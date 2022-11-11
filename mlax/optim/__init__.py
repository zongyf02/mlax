from mlax.optim import (
    sgd
)

from jax import (
    lax,
    tree_util
)

def apply_updates(
    update_gradients,
    model_weights,
    minimize = True
):
    """Apply update gradients to weights.

    :param update_gradients: Pytree of update gradients. Must be of the same
        tree structure as ``model_weights``.
    :param model_weights: Pytree of model weights.
    :param minimize: Whether to minimize or maximize loss.
        Default: True (minimize).

    :returns new_weights: New weights with the same pytree structure as
        ``model_weights``. ``model_weights`` with udpate gradients applied.
    """
    if minimize:
        fn = lambda g, w: lax.sub(w, g)
    else:
        fn = lambda g, w: lax.add(w, g)
    return tree_util.tree_map(fn, update_gradients, model_weights)
