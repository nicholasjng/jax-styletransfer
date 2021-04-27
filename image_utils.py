import os.path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from jax import tree_util


# TODO: Make target size a tuple for controlling aspect ratio
def load_image(fp: str, img_type: str, target_size: int = 512, dtype=None):
    if not os.path.exists(fp):
        raise ValueError(f"File {fp} does not exist.")

    print(f'Loading {img_type} image...')

    image = Image.open(fp)
    image = image.resize((target_size, target_size))
    image = jnp.array(image, dtype=dtype)
    image = image / 255.
    image = jnp.clip(image, 0., 1.)
    image = jnp.expand_dims(jnp.moveaxis(image, -1, 0), 0)

    print(f"{img_type.capitalize()} image loaded successfully. "
          f"Shape: {image.shape}")

    return image


def save_image(params: hk.Params, out_fp: str):
    im_data = tree_util.tree_leaves(params)[0]
    # clip values to avoid overflow problems in uint8 conversion
    im_data = jnp.clip(im_data, 0., 1.)

    # undo transformation block, squeeze off the batch dimension
    image: np.ndarray = np.squeeze(np.asarray(im_data))
    image = image * 255
    image = image.astype(np.uint8)
    image = np.moveaxis(image, 0, -1)

    # TODO: This needs to change for a tiled image
    im = Image.fromarray(image, mode="RGB")

    im.save(out_fp)


def checkpoint(params: hk.Params, out_dir: str, filename: str):
    """Saves the image at a checkpoint given by step."""
    out_fp = os.path.join(out_dir, filename)

    save_image(params, out_fp)
