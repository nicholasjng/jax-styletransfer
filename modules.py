from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp


class StyleLoss(hk.Module):

    def __init__(self, target, name: Optional[str] = None):
        super(StyleLoss, self).__init__(name=name)
        self.target_g = jax.lax.stop_gradient(gram_matrix(target))

    def __call__(self, x):
        g = gram_matrix(x)

        style_loss = jnp.mean(jnp.square(g - self.target_g))
        hk.set_state("style_loss", style_loss)

        return x


class ContentLoss(hk.Module):

    def __init__(self, target, name: Optional[str] = None):
        super(ContentLoss, self).__init__(name=name)
        self.target = jax.lax.stop_gradient(target)

    def __call__(self, x):
        content_loss = jnp.mean(jnp.square(x - self.target))
        hk.set_state("content_loss", content_loss)

        return x


def gram_matrix(x: jnp.array):
    """Computes Gram Matrix of an input array x."""
    # N-C-H-W format
    n, c, h, w = x.shape

    assert n == 1, "mini-batch has to be singular rn"

    features = jnp.reshape(x, (n * c, h * w))

    return jnp.dot(features, features.T) / (n * c * h * w)
