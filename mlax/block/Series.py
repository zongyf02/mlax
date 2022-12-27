import jax
from typing import Tuple, Union, Any
from functools import reduce
from collections import namedtuple
from mlax._utils import _get_fwd

def init(
    *layers: Tuple
) -> Tuple[Union[Tuple, jax.Array], Union[Tuple, jax.Array], Any]:
    """Initialize parameters and hyperparameters for a block that combines
    layers that do not require PRNGKeys in series.

    :param layers: Initialized parameters and hyperparameters from each of the
        layers.

    :returns trainables: Named tuple of trainable weights from each of the
        layers or trainables weights from the layer if there's only one layer.
    :returns non_trainables: Named tuple of non-trainable weights from each of
        the layers or non-trainables weights from the layer if there's only one
        layer.
    :returns hyperparams: Named tuple of hyperparams from each of the layers or
        hyperparams of the layer if there is only one layer.
    """
    if len(layers) <= 1:
        return layers[0]
    else:
        Series = namedtuple(
            "Series",
            (f"layer{i}" for i in range(len(layers)))
        )
        trainables, non_trainables, hyperparams = zip(*layers)
        return (
            Series(*trainables),
            Series(*non_trainables),
            Series(*hyperparams)
        )

def fwd(
    x: jax.Array,
    trainables: Union[Tuple, jax.Array],
    non_trainables: Union[Tuple, jax.Array],
    hyperparams: Any,
    inference_mode: bool=False
) -> Tuple[jax.Array, Tuple]:
    """Apply a series of layers that do not require PRNG keys.
    
    :param x: Input features.
    :param trainables: Named tuple of trainable weights from each of the
        layers or trainables weights from the layer if there's only one layer.
    :param non_trainables: Named tuple of non-trainable weights from each of
        the layers or non-trainables weights from the layer if there's only one
        layer.
    :param hyperparams: Named tuple of hyperparams from each of the layers or
        hyperparams of the layer if there is only one layer.
    :param inference_mode: Whether in inference or training mode. Default:
        False, training mode.

    :returns y: ``x`` with the layers applied in series.
    :returns non_trainables: Updated ``non_trainables``.
    """
    if isinstance(hyperparams, tuple):
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
            zip(trainables, non_trainables, hyperparams),
            x
        )
        return x, non_trainables.__class__(*new_ntrs)
    else:
        return _get_fwd(hyperparams)(
            x, trainables, non_trainables, hyperparams, inference_mode
        )