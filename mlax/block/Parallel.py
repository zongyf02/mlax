from jax import (
    tree_util
)
from typing import Tuple, Any
from collections import namedtuple
from mlax._utils import _get_fwd

def init(
    *layers
) -> Tuple[Tuple, Tuple, Any]:
    """Initialize parameters and hyperparameters for a block that combines
    layers that do not require PRNGKeys in parallel.

    :param layers: Initialized parameters and hyperparameters from each of the
        layers.

    :returns trainables: Named tuple of trainable weights from each of the
        layers.
    :returns non_trainables: Named tuple of non-trainable weights from each of
        the layers.
    :returns hyperparams: Named tuple of hyperparams from each of the layers.
    """
    Parallel = namedtuple(
        "Parallel",
        (f"layer{i}" for i in range(len(layers)))
    )
    trainables, non_trainables, hyperparams = zip(*layers)
    return (
        Parallel(*trainables),
        Parallel(*non_trainables),
        Parallel(*hyperparams)
    )

def fwd(
    x: Any,
    trainables: Tuple,
    non_trainables: Tuple,
    hyperparams: Any,
    inference_mode: bool=False
)  -> Tuple[Any, Tuple]:
    """Apply layers that do not require PRNG keys in parallel.

    :param x: PyTree of input features for each of the layers.
    :param trainables: Named tuple of trainable weights from each of the
        layers.
    :param non_trainables: Named tuple of non-trainable weights from each of
        the layers.
    :param hyperparams: Named tuple of hyperparams from each of the layers.
    :param inference_mode: Whether in inference or training mode. Default:
        False, training mode.

    :returns y: PyTree of ``x`` with layers applied.
    :returns non_trainables: Updated ``non_trainables`` from each of the layers.
    """
    x, treedef = tree_util.tree_flatten(x)

    def map_fn(param):
        x, tr, ntr, hp = param
        return _get_fwd(hp)(
            x, tr, ntr, hp, inference_mode
        )

    x, new_ntr = zip(*map(
        map_fn,
        zip(x, trainables, non_trainables, hyperparams)
    ))

    return (
        tree_util.tree_unflatten(treedef, x), non_trainables.__class__(*new_ntr)
    )
