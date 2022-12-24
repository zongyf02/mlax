import jax
from typing import Tuple, Union, Any
from functools import reduce
from mlax._utils import _get_fwd, _block_hyperparams

@_block_hyperparams
class SeriesHp:
    layers: Tuple

def init(
    *layers: Tuple
) -> Tuple[Union[Tuple, jax.Array], Union[Tuple, jax.Array], Any]:
    """Initialize parameters and hyperparameters for a layer that combines
    sub-layers that do not require PRNGKeys in series.

    :param layers: Initialized parameters and hyperparameters from each of the
        sub-layers.

    :returns trainables: Tuple of trainable weights from each of the sub-layer
        or trainables weights from the sub-layer if there's only one sub-layer.
    :returns non_trainables: Tuple of non-trainable weights from each of the 
        sub-layer or non-trainable weights from the sub-layer if there's only
        one sub-layer.
    :returns hyperparams: SeriesHp instance or hyperparam of the sub-layer if
        there is only one sub-layer.
    """
    if len(layers) <= 1:
        return layers[0]
    else:
        trainables, non_trainables, hyperparams = zip(*layers)
        return trainables, non_trainables, SeriesHp(hyperparams)

def fwd(
    x: jax.Array,
    trainables: Union[Tuple, jax.Array],
    non_trainables: Union[Tuple, jax.Array],
    hyperparams: Any,
    inference_mode: bool=False
) -> Tuple[jax.Array, Tuple]:
    """Apply a series of layers that do not require PRNG keys.
    
    :param x: Input features.
    :param trainables: Tuple of trainable weights from each of the layers or
        trainables weights from the layer if there's only one layer.
    :param non_trainables: Tuple of non-trainable weights from each of the
        layers or non-trainables weights from the layer if there's only one
        layer.
    :param hyperparams: SeriesHp instance or hyperparam of the layer if there is
        only one layer.
    :param inference_mode: Whether in inference or training mode. Default:
        False, training mode.

    :returns y: ``x`` with the layers applied in series.
    :returns non_trainables: Updated ``non_trainables``.
    """
    if not isinstance(hyperparams, SeriesHp):
        return _get_fwd(hyperparams)(
            x, trainables, non_trainables, hyperparams, inference_mode
        )
    else:
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
