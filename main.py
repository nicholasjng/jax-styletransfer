import copy
from typing import Mapping, Any

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image
from jax.flatten_util import ravel_pytree
from modules import StyleLoss, ContentLoss

Batch = Mapping[str, np.ndarray]
State = Mapping[str, jnp.ndarray]


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
                                       name=f"style_loss{s_losses}")
                layers.append(style_loss)

                s_losses += 1

            if name in style_layers:
                model.layers = tuple(layers)
                content_target = model(style_image)
                content_loss = ContentLoss(target=content_target,
                                           name=f"content_loss{c_losses}")
                layers.append(content_loss)
                c_losses += 1

            layers.append(jax.nn.relu)

    model.layers = tuple(layers)

    return model


def run_style_transfer(style_weight: float = 1e6,
                       content_weight: float = 1.,
                       num_steps: int = 300,
                       learning_rate: float = 1e-3):
    print('Loading content image...')
    content_image = Image.open('images/dancing.jpg')
    content_image = np.array(content_image).astype('float32') / 255.
    content_image = np.expand_dims(np.moveaxis(content_image, -1, 0), 0)
    print(f"Content image loaded successfully. Shape: {content_image.shape}")

    print('Loading content image...')
    style_image = Image.open('images/picasso-equal.jpg')
    style_image = np.array(style_image).astype('float32') / 255.
    style_image = np.expand_dims(np.moveaxis(style_image, -1, 0), 0)
    print(f"Content image loaded successfully. Shape: {style_image.shape}")

    # TODO: This sucks
    weight_list = [style_weight] * 3 + [content_weight] + [style_weight] * 2
    weight_array = jnp.array(weight_list)

    def net_fn(batch: Batch):
        x = batch
        vgg = build_augmented_vgg19(style_image=style_image,
                                    content_image=content_image)
        return vgg(x)

    def loss(current_params: hk.Params,
             current_state: State,
             current_batch: Batch):
        # calling convention for stateful transformation
        _, new_state = net.apply(current_params,
                                 current_state,
                                 rng,
                                 current_batch)

        loss_array, _ = ravel_pytree(new_state)

        return jnp.dot(loss_array, weight_array), new_state

    @jax.jit
    def update(current_params: hk.Params,
               current_opt_state: optax.OptState,
               current_state: State,
               current_batch: Batch):
        # -> Tuple[jnp.array, hk.Params, optax.OptState]:
        """Learning rule (stochastic gradient descent)."""
        (loss_val, new_state), grads = (
            jax.value_and_grad(loss, has_aux=True)(current_params,
                                                   current_state,
                                                   current_batch))

        updates, new_opt_state = opt.update(grads, current_opt_state,
                                            current_params)

        new_params = optax.apply_updates(params, updates)
        return loss_val, new_params, new_opt_state, new_state

    # We maintain the exponential moving average of the "live" params.
    # avg_params used only for evaluation (cf. https://doi.org/10.1137/0330046)
    @jax.jit
    def ema_update(p: Any, avg_p: Any):
        return optax.incremental_update(p, avg_p, step_size=0.001)

    net = hk.transform_with_state(net_fn)
    opt = optax.adam(learning_rate=learning_rate)
    rng = jax.random.PRNGKey(420)

    # Initialize network and optimiser; note we draw an input to get shapes.
    params, state = net.init(rng, style_image)

    avg_params = params

    opt_state = opt.init(params)

    input_img = copy.deepcopy(content_image)

    # Train/eval loop.
    for step in range(num_steps + 1):
        # Do SGD on a batch of training examples.
        losses, params, opt_state, state = update(params,
                                                  opt_state,
                                                  state,
                                                  input_img)
        avg_params = ema_update(params, avg_params)

        if step % 10 == 0:
            print(f"Losses: {losses}")
            print(f"State: {state}")


if __name__ == '__main__':
    run_style_transfer()
