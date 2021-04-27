import copy
from typing import Tuple, List
import time

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from image_utils import load_image, save_image
from models import augmented_vgg19
from modules import imagenet_mean, imagenet_std
from tree_utils import weighted_loss, calculate_losses, reduce_loss_tree

OptimizerUpdate = Tuple[hk.Params, optax.OptState, hk.State]


def run_style_transfer(content_fp: str,
                       style_fp: str,
                       model_fp: str,
                       content_weight: float = 1.,
                       style_weight: float = 1e6,
                       content_layers: List[str] = None,
                       style_layers: List[str] = None,
                       pooling: str = "avg",
                       num_steps: int = 300,
                       learning_rate: float = 1e-3,
                       save_image_every: int = 50):

    content_image = load_image(content_fp, "content")
    style_image = load_image(style_fp, "style")

    weights = {"content_loss": content_weight,
               "style_loss": style_weight}

    def net_fn(image: jnp.ndarray, is_training: bool = False):
        vgg = augmented_vgg19(fp=model_fp,
                              style_image=style_image,
                              content_image=content_image,
                              mean=imagenet_mean,
                              std=imagenet_std,
                              content_layers=content_layers,
                              style_layers=style_layers,
                              pooling=pooling)
        return vgg(image, is_training)

    def loss(trainable_params: hk.Params,
             non_trainable_params: hk.Params,
             current_state: hk.State,
             image: jnp.ndarray):

        merged_params = hk.data_structures.merge(trainable_params,
                                                 non_trainable_params)

        # stateful apply call, state contains the losses
        _, new_state = net.apply(merged_params, current_state,
                                 None, image, is_training=True)

        w_loss = weighted_loss(new_state, weights=weights)

        loss_val = reduce_loss_tree(w_loss)

        return loss_val, new_state

    @jax.jit
    def update(trainable_params: hk.Params,
               non_trainable_params: hk.Params,
               c_opt_state: optax.OptState,
               c_state: hk.State,
               image: jnp.ndarray) -> OptimizerUpdate:
        """Learning rule (stochastic gradient descent)."""
        (_, new_state), trainable_grads = (
            jax.value_and_grad(loss, has_aux=True)(trainable_params,
                                                   non_trainable_params,
                                                   c_state,
                                                   image))

        # update trainable params
        updates, new_opt_state = opt.update(trainable_grads,
                                            c_opt_state,
                                            trainable_params)

        new_params = optax.apply_updates(trainable_params, updates)

        return new_params, new_opt_state, new_state

    net = hk.transform_with_state(net_fn)
    opt = optax.adam(learning_rate=learning_rate)

    input_image = copy.deepcopy(content_image)

    # Initialize network and optimiser; we supply an input to get shapes.
    full_params, state = net.init(None, input_image, False)

    # split params into trainable and non-trainable
    t_params, nt_params = hk.data_structures.partition(
        lambda m, n, v: m == "norm",
        full_params
    )

    opt_state = opt.init(t_params)

    num_params = hk.data_structures.tree_size(full_params)
    num_t_params = hk.data_structures.tree_size(t_params)
    mem = hk.data_structures.tree_bytes(full_params)

    print(f"Total number of parameters: {num_params}")
    print(f"Number of trainable parameters: {num_t_params}")
    print(f"Number of non-trainable parameters: {num_params - num_t_params}")
    print(f"Memory footprint of network parameters: {mem / 1e6:.2f} MB")

    start = time.time()
    print("Starting style transfer optimization loop.")
    # Style transfer loop.
    # TODO: Think about changing to jax.lax control flow
    for step in range(num_steps + 1):
        # Do SGD on the same input image over and over again.
        t_params, opt_state, state = update(t_params, nt_params,
                                            opt_state, state, input_image)

        if step % 10 == 0:
            c_loss, s_loss = calculate_losses(state)

            print(f"Iteration: {step} Content loss: {c_loss:.4f} "
                  f"Style loss: {s_loss:.4f}")

        if step % save_image_every == 0:
            save_image(t_params, f"images/styled_it{step}.jpg")

    print(f"Style transfer finished. Took {(time.time() - start):.2f} secs.")


if __name__ == '__main__':
    run_style_transfer(
        content_fp="images/dancing.jpg",
        style_fp='images/picasso-equal.jpg',
        model_fp="models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",
        content_layers=['conv_4'],
        style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],
        pooling="avg",
        num_steps=20,
        learning_rate=1e-2,
        save_image_every=50)
