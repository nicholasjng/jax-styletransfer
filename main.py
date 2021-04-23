import copy
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from image_utils import load_image
from modules import StyleLoss, ContentLoss
from tree_utils import weighted_loss, calculate_losses, reduce_loss_tree


def build_augmented_vgg19(style_image: jnp.array,
                          content_image: jnp.array) -> hk.Sequential:
    layers = []
    n_conv, n_pools, c_losses, s_losses = 1, 1, 1, 1

    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
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


def run_style_transfer(style_weight: float = 1e6,
                       content_weight: float = 1.,
                       num_steps: int = 300,
                       learning_rate: float = 1e-3):
    content_image = load_image("images/dancing.jpg", "content")
    style_image = load_image('images/picasso-equal.jpg', "style")

    weights = {"content_loss": content_weight,
               "style_loss": style_weight}

    def net_fn(batch: jnp.ndarray):
        x = batch
        vgg = build_augmented_vgg19(style_image=style_image,
                                    content_image=content_image)
        return vgg(x)

    def loss(current_params: hk.Params,
             current_state: hk.State,
             current_batch: jnp.ndarray):
        # calling convention for stateful transformation
        # state contains the losses
        _, new_state = net.apply(current_params,
                                 current_state,
                                 rng,
                                 current_batch)

        w_loss = weighted_loss(new_state, weights=weights)

        loss_val = reduce_loss_tree(w_loss)

        return loss_val, new_state

    @jax.jit
    def update(current_params: hk.Params,
               current_opt_state: optax.OptState,
               current_state: hk.State,
               current_batch: jnp.ndarray) \
            -> Tuple[hk.Params, optax.OptState, hk.State]:
        """Learning rule (stochastic gradient descent)."""
        (_, new_state), grads = (
            jax.value_and_grad(loss, has_aux=True)(current_params,
                                                   current_state,
                                                   current_batch))

        updates, new_opt_state = opt.update(grads,
                                            current_opt_state,
                                            current_params)

        new_params = optax.apply_updates(current_params, updates)

        return new_params, new_opt_state, new_state

    net = hk.transform_with_state(net_fn)
    opt = optax.adam(learning_rate=learning_rate)
    rng = jax.random.PRNGKey(420)

    # Initialize network and optimiser; note we draw an input to get shapes.
    params, state = net.init(rng, style_image)
    opt_state = opt.init(params)

    input_img = copy.deepcopy(content_image)

    # Training loop.
    for step in range(num_steps + 1):
        # Do SGD on a batch of training examples.
        params, opt_state, state = update(params, opt_state, state, input_img)

        if step % 10 == 0:
            c_loss, s_loss = calculate_losses(state)

            print(f"Content loss: {c_loss:.4f} Style loss: {s_loss:.4f}")


if __name__ == '__main__':
    run_style_transfer()
