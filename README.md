# jax-styletransfer
Neural Style transfer, implemented in the JAX ecosystem.


This repository houses a small project about Neural Style Transfer in JAX. I started it because:
1. I want to get more experience with JAX and the ML/DL ecosystem around it (including, but not limited to Haiku, Optax, Flax, Pyro etc.)
2. My end goal is to style high resolution images from my camera (24 Mpx RAW/JPEG files). This would have to work as a batch inference job via tiling the image, because single graphics cards cannot handle convolutions of these insane image sizes just yet (I think?).

In the end, if this code becomes fast/maintainable enough, it might even be portable / deployable as a ML microservice. Let's see how it turns out :-)

## Requirements
Requirements are DeepMind's Haiku, Optax, as well as JAX itself (pinned to latest versions as of April 20th). Numpy and others are not listed because they are obligatory for any JAX code by proxy.

The algorithm is adapted from the example in https://pytorch.org/tutorials/advanced/neural_style_tutorial.html.
