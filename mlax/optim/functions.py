from jax import lax
from jax import tree_util

def apply(
    gradients,
    weights,
    minimize = True
):
    """Apply gradients to weights.

    :param gradients: Pytree of gradients. Must be of the same tree structure
        as ``weights``.
    :param weights: Pytree of weights.
    :param minimize: Whether to minimize or maximize loss.
        Default: True (minimize).

    :returns new_weights: New weights of the same shape as ``weights``, after
        gradients are applied
    """
    if minimize:
        fn = lambda g, w: lax.sub(w, g)
    else:
        fn = lambda g, w: lax.add(w, g)
    return tree_util.tree_map(fn, gradients, weights)
