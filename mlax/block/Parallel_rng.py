from jax import (
    tree_util,
    random
)
from functools import reduce
from typing import Tuple, Any, NamedTuple
from mlax._utils import _get_fwd, _needs_key

class Hyperparams(NamedTuple):
    layers: Tuple

def init(
    *layers
) -> Tuple[Tuple, Tuple, Hyperparams]:
    """Initialize parameters and hyperparameters for a layer that combines
    layers that may require PRNGKeys in parallel.

    :param layers: Initialized parameters and hyperparameters from each of the
        sub-layers.

    :returns trainables: Tuple of trainable weights from each of the sub-layers.
    :returns non_trainables: Tuple of non-trainable weights from each of the 
        sub-layers.
    :returns hyperparams: NamedTuple containing the hyperparameters.
    """
    trainables, non_trainables, hyperparams = zip(*layers)
    return trainables, non_trainables, Hyperparams(hyperparams)

def fwd(
    x: Any,
    trainables: Tuple[Any],
    non_trainables: Tuple[Any],
    key: Any,
    hyperparams: Tuple[Any],
    inference_mode: bool=False
):
    """Apply layers that may require PRNG keys in parallel.

    :param x: PyTree of input features for each of the layers.
    :param trainables: Tuple of trainable weights from each of the layers.
    :param non_trainables: Tuple of non-trainable weights from each of the
        layers.
    :param key: PRNG key.
    :param hyperparams: NamedTuple containing the hyperparameters.
    :param inference_mode: Whether in inference or training mode. Default:
        False, training mode.

    :returns y: PyTree of ``x`` with layers applied.
    :returns non_trainables: Updated ``non_trainables`` from each of the layer.
    """
    x, treedef = tree_util.tree_flatten(x)

    fwds = []
    needs_keys = []
    def reduce_fn(accum, hp):
        fwd = _get_fwd(hp)
        needs_key = _needs_key(fwd)
        fwds.append(fwd)
        needs_keys.append(needs_key)
        return accum + needs_key

    n_keys = reduce(reduce_fn, hyperparams.layers, 0)
    if n_keys > 1:
        keys_iter = iter(random.split(key, n_keys))
    else:
        keys_iter = iter((key,))

    def map_fn(param):
        x, fwd, needs_key, tr, ntr, hp = param
        if needs_key:
            return fwd(
                x, tr, ntr, next(keys_iter), hp, inference_mode
            )
        else:
            return fwd(
                x, tr, ntr, hp, inference_mode
            )

    x, non_trainables = zip(*map(
        map_fn,
        zip(x, fwds, needs_keys, trainables, non_trainables, hyperparams.layers)
    ))

    return tree_util.tree_unflatten(treedef, x), non_trainables
