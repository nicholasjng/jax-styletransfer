import jax
from typing import Optional, List, Tuple
import jax.numpy as jnp
import haiku as hk
import optax


class StyleLoss(hk.Module):

    def __init__(self, target, name: Optional[str] = None):
        super(StyleLoss, self).__init__(name=name)
        self.target_g = jax.lax.stop_gradient(gram_matrix(target))
        self.loss = None

    def __call__(self, x):
        g = gram_matrix(x)
        self.loss = jnp.mean(jnp.square(g - self.target_g))
        return x


class ContentLoss(hk.Module):
    
    def __init__(self, target, name: Optional[str] = None):
        super(ContentLoss, self).__init__(name=name)
        self.target = jax.lax.stop_gradient(target)
        self.loss = None

    def __call__(self, x):
        self.loss = jnp.mean(jnp.square(x - self.target))


def gram_matrix(x: jnp.array):
    """Computes Gram Matrix of an input array x"""
    a, b, c, d = x.shape

    # TODO: This transpose needs more thought
    return jnp.dot(x, x.T) / (b * c * d)


def make_optimizer(lr: float) -> optax.GradientTransformation:
    return optax.adam(learning_rate=lr)


def build_shallow_vgg19(config: List[Tuple],
                        style_image: jnp.array,
                        content_image: jnp.array,
                        ) -> hk.Sequential:
    layers = []
    n_convs, n_pools, c_losses, s_losses = 1, 1, 1, 1

    model = hk.Sequential(layers=layers)

    for tup in config:
        kind, kwargs = tup
        if kind == "max_pool":
            layers.append(hk.MaxPool(**kwargs,
                                     #window_shape=2,
                                     #strides=2,
                                     #padding="VALID",
                                     name=f"max_pool_{n_pools}"))
            n_pools += 1
        elif kind == "conv2d":
            layers.append(hk.Conv2D(**kwargs,
                                    # kernel_shape=3,
                                    # stride=1,
                                    # padding="SAME",
                                    name=f"conv2d_{n_convs}"))

            n_convs += 1

        elif kind == "style":
            model.layers = tuple(layers)
            style_target = model(content_image)
            style_loss = StyleLoss(target=style_target,
                                   name=f"style_loss{s_losses}")
            layers.append(style_loss)
            s_losses += 1

        elif kind == "content":
            model.layers = tuple(layers)
            content_target = model(style_image)
            content_loss = ContentLoss(target=content_target,
                                       name=f"content_loss{c_losses}")
            layers.append(content_loss)
            c_losses += 1

    model.layers = tuple(layers)

    return model


if __name__ == '__main__':
    print("Hello World")
