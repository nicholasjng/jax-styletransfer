from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp

# ImageNet statistics
imagenet_mean = jnp.array([0.485, 0.456, 0.406])
imagenet_std = jnp.array([0.229, 0.224, 0.225])


def gram_matrix(x: jnp.ndarray):
    """Computes Gram Matrix of an input array x."""
    # N-C-H-W format
    # TODO: Refactor this to compute the Gram matrix of a feature map
    #  and then apply jax.vmap on the batch dimension
    n, c, h, w = x.shape

    assert n == 1, "mini-batch has to be singular right now"

    features = jnp.reshape(x, (n * c, h * w))

    return jnp.dot(features, features.T) / (n * c * h * w)


class StyleLoss(hk.Module):
    """Identity layer capturing the style loss between input and target."""

    def __init__(self, target, name: Optional[str] = None):
        super(StyleLoss, self).__init__(name=name)
        self.target_g = jax.lax.stop_gradient(gram_matrix(target))

    def __call__(self, x):
        g = gram_matrix(x)

        style_loss = jnp.mean(jnp.square(g - self.target_g))
        hk.set_state("style_loss", style_loss)

        return x


class ContentLoss(hk.Module):
    """Identity layer capturing the content loss between input and target."""

    def __init__(self, target, name: Optional[str] = None):
        super(ContentLoss, self).__init__(name=name)
        self.target = jax.lax.stop_gradient(target)

    def __call__(self, x):
        content_loss = jnp.mean(jnp.square(x - self.target))
        hk.set_state("content_loss", content_loss)

        return x


class Normalization(hk.Module):
    # create a module normalizing the input image
    # so we can easily put it into a hk.Sequential
    def __init__(self,
                 image: jnp.ndarray,
                 mean: jnp.ndarray,
                 std: jnp.ndarray,
                 name: Optional[str] = None):
        super(Normalization, self).__init__(name=name)

        # save image to make it a trainable parameter
        self.image = image

        # reshape mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [N x C x H x W].
        self.mean = jnp.array(mean).reshape(-1, 1, 1)
        self.std = jnp.array(std).reshape(-1, 1, 1)

    def forward(self, x: jnp.ndarray):
        # throw away the input and (re-)use the tracked parameter instead
        # this assures that the image is actually styled
        img = hk.get_parameter("image",
                               shape=self.image.shape,
                               dtype=self.image.dtype,
                               init=hk.initializers.Constant(self.image))

        return (img - self.mean) / self.std
