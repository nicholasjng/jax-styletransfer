import copy
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from image_utils import load_image
from models import augmented_vgg19
from tree_utils import weighted_loss, calculate_losses, reduce_loss_tree


def run_style_transfer(style_weight: float = 1e6,
                       content_weight: float = 1.,
                       num_steps: int = 300,
                       learning_rate: float = 1e-3):
    content_image = load_image("images/dancing.jpg", "content")
    style_image = load_image('images/picasso-equal.jpg', "style")

    weights = {"content_loss": content_weight,
               "style_loss": style_weight}

    def net_fn(image: jnp.ndarray):
        vgg = augmented_vgg19(style_image=style_image,
                              content_image=content_image)
        return vgg(image)

    def loss(trainable_params: hk.Params,
             non_trainable_params: hk.Params,
             state: hk.State,
             image: jnp.ndarray):

        merged_params = hk.data_structures.merge(trainable_params,
                                                 non_trainable_params)

        # stateful apply call, state contains the losses
        _, new_state = net.apply(merged_params, state, rng, image)

        w_loss = weighted_loss(new_state, weights=weights)

        loss_val = reduce_loss_tree(w_loss)

        return loss_val, new_state

    @jax.jit
    def update(trainable_params: hk.Params, non_trainable_params: hk.Params,
               c_opt_state: optax.OptState, c_state: hk.State,
               image: jnp.ndarray) \
            -> Tuple[hk.Params, optax.OptState, hk.State]:
        """Learning rule (stochastic gradient descent)."""
        (_, new_state), trainable_grads = (
            jax.value_and_grad(loss, has_aux=True)(trainable_params,
                                                   non_trainable_params,
                                                   c_state,
                                                   image))

        # update trainable params
        updates, new_opt_state = opt.update(trainable_grads, c_opt_state)

        new_params = optax.apply_updates(trainable_params, updates)

        return new_params, new_opt_state, new_state

    net = hk.transform_with_state(net_fn)
    opt = optax.adam(learning_rate=learning_rate)
    rng = jax.random.PRNGKey(420)

    # Initialize network and optimiser; we supply an input to get shapes.
    full_params, state = net.init(rng, style_image)
    opt_state = opt.init(full_params)

    input_img = copy.deepcopy(content_image)

    t_params, nt_params = hk.data_structures.partition(
        lambda m, n, v: m == "normalization",
        full_params
    )

    # Training loop.
    for step in range(num_steps + 1):
        # Do SGD on a batch of training examples.
        t_params, opt_state, state = update(t_params, nt_params,
                                            opt_state, state, input_img)

        if step % 10 == 0:
            c_loss, s_loss = calculate_losses(state)

            print(f"Content loss: {c_loss:.4f} Style loss: {s_loss:.4f}")


if __name__ == '__main__':
    run_style_transfer()
