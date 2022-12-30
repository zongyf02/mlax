import jax
from jax import (
    nn,
    lax,
    random
)
from math import prod
from typing import Tuple, Any
from mlax._utils import (
    _nn_hyperparams,
    _n_elems,
    _mean,
    _variance,
    _normalize
)

@_nn_hyperparams
class BatchNormHp:
    channel_last: bool
    epsilon: Any
    momentum: Any

def init(
    key,
    in_channels: int,
    channel_last: bool=False,
    epsilon=1e-05,
    momentum=0.9,
    mean_initializer=nn.initializers.zeros,
    var_initializer=nn.initializers.ones,
    dtype=None
) -> Tuple[None, Tuple[jax.Array, jax.Array], BatchNormHp]:
    """Initialize parameters and hyperparameters for a batch norm layer.

    :param key: PRNG key for weight initialization.
    :param in_channels: Number of input feature dimensions/channels.
    :param channel_last: Whether features are channel-last or first. Default:
        False, channel-first.
    :param epsilon: Small number added to variance to avoid divisions by zero.
        Default: 1e-05.
    :param momemtum: Momentum for the moving average.
    :param mean_initializer: Moving mean initializer as defined by
        ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
        Default:: zeros.
    :param var_initializer: moving variance initializer as defined by
        ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
        Default:: ones.
    :param dtype: Type of initialized moving mean and variance weight. Default:
        None. ``mean_initializer`` and ``var_initializer``'s default.

    :returns trainables: None.
    :returns non_trainables: Initialized moving average and variance.
    :returns hyperparams: BatchNormHp instance.
    """
    key1, key2 = random.split(key)
    moving_mean = mean_initializer(key1, (in_channels,), dtype)
    moving_var = var_initializer(key2, (in_channels,), dtype)

    return None, (moving_mean, moving_var), BatchNormHp(
        channel_last,
        epsilon,
        momentum
    )

def fwd(
    x: jax.Array,
    trainables: None,
    non_trainables: Tuple[jax.Array, jax.Array],
    hyperparams: BatchNormHp,
    inference_mode: bool=False
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
    """Apply batch normalization without the learnable parameters.
 
    :param x: Batched input features to the batch norm. Must be of ``dtype`` and
        compatible with ``channel_last``.
    :param trainables: Trainable weights for a batch norm. Should be None.
        Ignored.
    :param non_trainables: Non-trainable weights for a batch norm.
    :param hyperparams: BatchNormHp instance.
    :param inference_mode: Whether in inference or training mode. If in
        inference mode, the moving mean and variance are used to normalize input
        features. If in training mode, the batch mean and variance are used, and
        the moving mean and variance are updated. Default: False, training mode.
 
    :returns y: Batch normalized ``x``.
    :returns non_trainables: Updated non-trainables.
    """
    _ndims = len(x.shape)
    if inference_mode:
        mean, var = non_trainables
        mean = lax.convert_element_type(mean, x.dtype)
        var = lax.convert_element_type(var, x.dtype)

        if hyperparams.channel_last:
            broadcast_dims = (_ndims - 1,)
        else:
            broadcast_dims = (1,)
    else:
        # Compute mean and variance
        if hyperparams.channel_last:
            reduce_dims = range(0, _ndims - 1)
            broadcast_dims = (_ndims - 1,)
        else:
            reduce_dims = (0, *range(2, _ndims))
            broadcast_dims = (1,)

        n_elems = _n_elems(x, reduce_dims)
        mean = _mean(x, reduce_dims, n_elems)
        var = _variance(x, reduce_dims, n_elems, mean)

        # Update non_trainables
        moving_mean, moving_var = non_trainables
        momentum = lax.convert_element_type(
            hyperparams.momentum, moving_mean.dtype
        )
        one_m_momentum = lax.convert_element_type(
            1.0 - hyperparams.momentum, moving_mean.dtype
        )
        moving_mean = lax.add(
            moving_mean * momentum,
            lax.convert_element_type(mean, moving_mean.dtype) * one_m_momentum
        )
        moving_var = lax.add(
            moving_var * momentum,
            lax.convert_element_type(var, moving_var.dtype) * one_m_momentum
        )
        non_trainables = (moving_mean, moving_var)

    return _normalize(
        x,
        broadcast_dims,
        hyperparams.epsilon,
        mean,
        var
    ), non_trainables
