from jax import (
    tree_util
)
from typing import Tuple, Any, NamedTuple
from mlax._utils import _get_fwd

class Hyperparams(NamedTuple):
    layers: Tuple

def init(
    *layers
) -> Tuple[Tuple, Tuple, Hyperparams]:
    """Initialize parameters and hyperparameters for a layer that combines
    layers that do not require PRNGKeys in parallel.

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
    hyperparams: Tuple[Any],
    inference_mode: bool=False
):
    """Apply layers that do not require PRNG keys in parallel.

    :param x: PyTree of input features for each of the layers.
    :param trainables: Tuple of trainable weights from each of the layers.
    :param non_trainables: Tuple of non-trainable weights from each of the
        layers.
    :param hyperparams: NamedTuple containing the hyperparameters.
    :param inference_mode: Whether in inference or training mode. Default:
        False, training mode.

    :returns y: PyTree of ``x`` with layers applied.
    :returns non_trainables: Updated ``non_trainables`` from each of the layer.
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
