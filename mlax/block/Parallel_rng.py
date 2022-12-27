from jax import (
    tree_util,
    random
)
from typing import Tuple, Any
from collections import namedtuple
from mlax._utils import _get_fwd, _needs_key

def init(
    *layers
) -> Tuple[Tuple, Tuple, Any]:
    """Initialize parameters and hyperparameters for a block that combines
    layers that may require PRNGKeys in parallel.

    :param layers: Initialized parameters and hyperparameters from each of the
        layers.

    :returns trainables: Named tuple of trainable weights from each of the
        layers.
    :returns non_trainables: Named tuple of non-trainable weights from each of
        the layers.
    :returns hyperparams: Named tuple of hyperparams from each of the layers.
    """
    ParallelRng = namedtuple(
        "ParallelRng",
        (f"layer{i}" for i in range(len(layers)))
    )
    trainables, non_trainables, hyperparams = zip(*layers)
    return (
        ParallelRng(*trainables),
        ParallelRng(*non_trainables),
        ParallelRng(*hyperparams)
    )

def fwd(
    x: Any,
    trainables: Tuple,
    non_trainables: Tuple,
    key: Any,
    hyperparams: Any,
    inference_mode: bool=False
)  -> Tuple[Any, Tuple]:
    """Apply layers that may require PRNG keys in parallel.

    :param x: PyTree of input features for each of the layers.
    :param trainables: Named tuple of trainable weights from each of the
        layers.
    :param non_trainables: Named tuple of non-trainable weights from each of
        the layers.
    :param key: PRNG key.
    :param hyperparams: Named tuple of hyperparams from each of the layers.
    :param inference_mode: Whether in inference or training mode. Default:
        False, training mode.

    :returns y: PyTree of ``x`` with layers applied.
    :returns non_trainables: Updated ``non_trainables`` from each of the layers.
    """
    x, treedef = tree_util.tree_flatten(x)

    fwds = tuple(map(_get_fwd, hyperparams))
    needs_keys = tuple(map(_needs_key, fwds))
    n_keys = sum(needs_keys)
    if n_keys > 1:
        keys_iter = iter(random.split(key, n_keys))
    else:
        keys_iter = iter((key,))

    def map_fn(param):
        x, tr, ntr, hp, fwd, needs_key = param
        if needs_key:
            return fwd(
                x, tr, ntr, next(keys_iter), hp, inference_mode
            )
        else:
            return fwd(
                x, tr, ntr, hp, inference_mode
            )

    x, new_ntr = zip(*map(
        map_fn,
        zip(
            x, trainables, non_trainables, hyperparams, fwds, needs_keys
        )
    ))

    return (
        tree_util.tree_unflatten(treedef, x), non_trainables.__class__(*new_ntr)
    )
