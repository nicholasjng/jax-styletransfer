from typing import Any

import jax.numpy as jnp
from PIL import Image


def load_image(fp: str, img_type: str, dtype: Any = None):
    print(f'Loading {img_type} image...')

    image = Image.open(fp)
    image = jnp.array(image, dtype=dtype) / 255.
    image = jnp.expand_dims(jnp.moveaxis(image, -1, 0), 0)

    print(f"{img_type.capitalize()} image loaded successfully. "
          f"Shape: {image.shape}")

    return image
