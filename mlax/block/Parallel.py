from jax import (
    tree_util
)
from typing import Tuple, Any, NamedTuple
from mlax._utils import _get_fwd, _block_hyperparams

@_block_hyperparams
class ParallelHp:
    layers: Tuple

def init(
    *layers
) -> Tuple[Tuple, Tuple, ParallelHp]:
    """Initialize parameters and hyperparameters for a layer that combines
    sub-layers that do not require PRNGKeys in parallel.

    :param layers: Initialized parameters and hyperparameters from each of the
        sub-layers.

    :returns trainables: Tuple of trainable weights from each of the sub-layers.
    :returns non_trainables: Tuple of non-trainable weights from each of the 
        sub-layers.
    :returns hyperparams: ParallelHp instance.
    """
    trainables, non_trainables, hyperparams = zip(*layers)
    return trainables, non_trainables, ParallelHp(hyperparams)

def fwd(
    x: Any,
    trainables: Tuple,
    non_trainables: Tuple,
    hyperparams: ParallelHp,
    inference_mode: bool=False
)  -> Tuple[Any, Tuple]:
    """Apply layers that do not require PRNG keys in parallel.

    :param x: PyTree of input features for each of the layers.
    :param trainables: Tuple of trainable weights from each of the layers.
    :param non_trainables: Tuple of non-trainable weights from each of the
        layers.
    :param hyperparams: ParallelHp instance.
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

    x, non_trainables = zip(*map(
        map_fn,
        zip(x, trainables, non_trainables, hyperparams.layers)
    ))

    return tree_util.tree_unflatten(treedef, x), non_trainables
