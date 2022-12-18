import jax
from typing import Tuple, Any, NamedTuple
from functools import reduce
from mlax._utils import _get_fwd

class Hyperparams(NamedTuple):
    layers: Tuple

def init(
    *layers: Tuple
) -> Tuple[Tuple, Tuple, Hyperparams]:
    """Initialize parameters and hyperparameters for a layer that combines
    sub-layers that do not require PRNGKeys in series.

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
    x: jax.Array,
    trainables: Tuple[Any],
    non_trainables: Tuple[Any],
    hyperparams: Hyperparams,
    inference_mode: bool=False
) -> Tuple[jax.Array, Tuple]:
    """Apply a series of layers that do not require PRNG keys.
    
    :param x: Input features.
    :param trainables: Tuple of trainable weights from each of the layers.
    :param non_trainables: Tuple of non-trainable weights from each of the
        layers.
    :param hyperparams: NamedTuples containing the hyperparameters.
    :param inference_mode: Whether in inference or training mode. Default:
        False, training mode.

    :returns y: ``x`` with the layers applied in series.
    :returns non_trainables: Updated ``non_trainables`` from each of the layer.
    """
    new_ntrs=[]
    def reduce_fn(x, params):
        tr, ntr, hp = params
        x, new_ntr = _get_fwd(hp)(
            x, tr, ntr, hp, inference_mode
        )
        new_ntrs.append(new_ntr)
        return x

    x = reduce(
        reduce_fn,
        zip(trainables, non_trainables, hyperparams.layers),
        x
    )
    return x, tuple(new_ntrs)
