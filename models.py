from typing import Dict, List

import h5py
import haiku as hk
import jax
import jax.numpy as jnp

from modules import StyleLoss, ContentLoss, Normalization

__all__ = ["augmented_vgg19",
           "augmented_inception_v3"]

Parameters = Dict[str, Dict[str, jnp.ndarray]]


def get_model_params(fp: str, verbose: bool = False) -> Parameters:
    """
    Utility for extracting layers from a h5 file.

    Designed to work with model files downloaded from
    https://github.com/fchollet/deep-learning-models/.
    """
    hf = h5py.File(fp, 'r')

    config = dict()

    # layers (weights + biases) arranged in h5.Groups
    for layer_name, group in hf.items():
        param_dict = dict()
        print(f"Layer name: {layer_name}")
        for k, v in group.items():
            values = jnp.array(v)
            # add two dummy dims to bias layers
            if len(values.shape) == 1:
                values.reshape((-1, 1, 1))

            if verbose:
                print(f"Parameter name: {k} Parameter shape: {v.shape}")

            # give weights / biases special names in param dict
            if "w" in k.lower():
                transformed_key = "w"
            elif "b" in k.lower():
                transformed_key = "b"
            else:
                transformed_key = k

            param_dict[transformed_key] = values

        config[layer_name] = param_dict

    return config


def augmented_vgg19(fp: str,
                    content_image: jnp.ndarray,
                    style_image: jnp.ndarray,
                    mean: jnp.ndarray,
                    std: jnp.ndarray,
                    content_layers: List[str] = None,
                    style_layers: List[str] = None,
                    pooling: str = "avg") -> hk.Sequential:
    """Build a VGG19 network augmented by content and style loss layers."""

    if pooling.lower() not in ["avg", "max"]:
        raise ValueError("Pooling method not recognized")

    params = get_model_params(fp=fp)

    # prepend a normalization layer
    layers = [Normalization(content_image, mean, std, "norm")]

    n_conv, n_pools, c_losses, s_losses = 1, 1, 1, 1

    # desired depth layers to compute style/content losses :
    content_layers = content_layers or []
    style_layers = style_layers or []

    model = hk.Sequential(layers=layers)

    for k, p_dict in params:
        if "pool" in k:
            if pooling == "avg":
                layers.append(hk.AvgPool(window_shape=2,
                                         strides=2,
                                         padding="VALID",
                                         channel_axis=1,
                                         name=f"avg_pool_{n_pools}"))
            else:
                layers.append(hk.MaxPool(window_shape=2,
                                         strides=2,
                                         padding="VALID",
                                         channel_axis=1,
                                         name=f"max_pool_{n_pools}"))
            n_pools += 1

        elif "conv" in k:
            name = f"conv_{n_conv}"

            in_ch, kernel_h, kernel_w, out_ch = p_dict["w"].shape

            kernel_shape = (kernel_h, kernel_w)

            layers.append(hk.Conv2D(
                output_channels=out_ch,
                kernel_shape=kernel_shape,
                stride=1,
                padding="SAME",
                data_format="NCHW",
                w_init=hk.initializers.Constant(p_dict["w"]),
                b_init=hk.initializers.Constant(p_dict["b"]),
                name=name))

            n_conv += 1

            if name in content_layers:
                model.layers = tuple(layers)
                style_target = model(content_image)
                layers.append(StyleLoss(target=style_target,
                                        name=f"style_loss_{s_losses}"))
                s_losses += 1

            if name in style_layers:
                model.layers = tuple(layers)
                content_target = model(style_image)
                layers.append(ContentLoss(target=content_target,
                                          name=f"content_loss_{c_losses}"))
                c_losses += 1

            layers.append(jax.nn.relu)

    model.layers = tuple(layers)

    return model


def augmented_inception_v3():
    pass
