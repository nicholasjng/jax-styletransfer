from typing import Any, Mapping

import jax.numpy as jnp
from jax import tree_util
import haiku as hk

__all__ = ["reduce_loss_tree",
           "weighted_loss",
           "split_loss_tree",
           "calculate_losses"]


def reduce_loss_tree(loss_tree: Mapping) -> jnp.array:
    """Reduces a loss tree to a scalar (i.e. jnp.array w/ size 1)."""
    return tree_util.tree_reduce(lambda x, y: x + y, loss_tree)


def weighted_loss(loss_tree: Mapping, weights: Mapping) -> Any:
    """Updates a loss tree by applying weights at the leaves."""
    return hk.data_structures.map(
            # m: module_name, n: param name, v: param value
            lambda m, n, v: weights[n] * v,
            loss_tree)


def split_loss_tree(loss_tree: Mapping):
    """Splits a loss tree into content and style loss trees."""
    return hk.data_structures.partition(
        lambda m, n, v: n == "content_loss",
        loss_tree)


def calculate_losses(loss_tree: Mapping):
    """Returns a tuple of current content loss and style loss."""
    # obtain content and style trees
    c_tree, s_tree = split_loss_tree(loss_tree)

    # reduce and return
    return reduce_loss_tree(c_tree), reduce_loss_tree(s_tree)
