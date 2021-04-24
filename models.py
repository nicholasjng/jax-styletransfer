# import h5py

# import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp

from modules import StyleLoss, ContentLoss

__all__ = ["augmented_vgg19",
           "augmented_inception_v3"]


def augmented_model():
    pass


def augmented_vgg19(style_image: jnp.ndarray,
                    content_image: jnp.ndarray) -> hk.Sequential:
    layers = []
    n_conv, n_pools, c_losses, s_losses = 1, 1, 1, 1

    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
           'M',
           512, 512, 512, 'M']

    # desired depth layers to compute style/content losses :
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    model = hk.Sequential(layers=layers)

    for val in cfg:
        if val == "M":
            layers.append(hk.MaxPool(window_shape=2,
                                     strides=2,
                                     padding="VALID",
                                     channel_axis=1,
                                     name=f"max_pool_{n_pools}"))
            n_pools += 1

        else:
            name = f"conv_{n_conv}"
            layers.append(hk.Conv2D(output_channels=val,
                                    kernel_shape=3,
                                    stride=1,
                                    padding="SAME",
                                    data_format="NCHW",
                                    name=name))
            n_conv += 1

            if name in content_layers:
                model.layers = tuple(layers)
                style_target = model(content_image)
                style_loss = StyleLoss(target=style_target,
                                       name=f"style_loss_{s_losses}")
                layers.append(style_loss)

                s_losses += 1

            if name in style_layers:
                model.layers = tuple(layers)
                content_target = model(style_image)
                content_loss = ContentLoss(target=content_target,
                                           name=f"content_loss_{c_losses}")
                layers.append(content_loss)
                c_losses += 1

            layers.append(jax.nn.relu)

    model.layers = tuple(layers)

    return model


def augmented_inception_v3():
    pass
