# jax-styletransfer

Neural Style transfer, implemented in the JAX ecosystem.

This repository houses a small project about Neural Style Transfer in JAX. I started it because:

1. I want to get more experience with JAX and the ML/DL ecosystem around it (including, but not limited to Haiku, Optax,
   Flax, Pyro etc.)
2. My end goal is to style high resolution images from my camera (24 Mpx RAW/JPEG files). This would have to work as a
   batch inference job via tiling the image, because single graphics cards cannot handle convolutions of these insane
   image sizes just yet (I think?).

In the end, if this code becomes fast/maintainable enough, it might even be portable / deployable as a ML microservice.
Let's see how it turns out :-)

## Quickstart

In a virtual environment, follow the steps below to run a style transfer using a pretrained VGG19 on two Pytorch
tutorial images.

```
# staging
git clone https://github.com/njunge94/jax-styletransfer.git
cd jax-styletransfer
pip install --upgrade pip
pip install -r requirements.txt
# for CPU-only installation of JAX
pip install --upgrade jax jaxlib
# OR: JAX for CUDA linked against cuda version X.X:
pip install --upgrade jax jaxlib==0.1.65+cudaX.X -f https://storage.googleapis.com/jax-releases/jax_releases.html


# save images and model weights into their own folder
mkdir images
wget -P images -i test-files.txt
mkdir models
wget -P models https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5

python main.py images/dancing.jpg images/picasso.jpg models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
```

For a detailed list of optional command line flags, check out the source code in `main.py`.

## Requirements

Requirements are DeepMind's Haiku, Optax, as well as JAX itself (pinned to latest versions as of April 20th). Numpy and
others are not listed because they are obligatory for any JAX code by proxy. Two additional requirements are PIL for
image loading and saving, and h5py for loading precomputed weights.

The algorithm is adapted from the example in https://pytorch.org/tutorials/advanced/neural_style_tutorial.html.

## TODOs

* [ ] Write utilities for tiling of high resolution images
* [ ] Write a Dockerfile
* [ ] Add more pretrained models (Inception v3, ResNet etc.)
* [ ] Add tests
* [x] Add absl flags for command line invocation
* [x] Finish the training loop
* [x] Add loss and normalization modules with hk.State
* [x] Implement image loading / saving and transformations
* [x] Implement weight loading from HDF5 and augmented VGG19

## References and acknowledgements

I used the following sources to implement this project:

* Original paper: "[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)" by Leon A. Gatys, Alexander
  S. Ecker and Matthias Bethge
* [Neural transfer tutorial in Pytorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) (based on
  VGG19 as well)
* A really
  helpful [tutorial on loading pretrained models in Haiku](https://www.pragmatic.ml/finetuning-transformers-with-jax-and-haiku/)
  by Madison May, released on Pragmatic ML
* Weights for pretrained models were sourced from F.
  Chollet's [Deep Learning Models GitHub repository](https://github.com/fchollet/deep-learning-models/releases).

If you are also learning the JAX ecosystem, the packages' respective documentations might be helpful to you as well:

* [JAX documentation](https://jax.readthedocs.io/en/latest/)
  , [Haiku documentation](https://dm-haiku.readthedocs.io/en/latest/index.html)
  , [Optax documentation](https://optax.readthedocs.io/en/latest/)
