import jax
from jax import (
    nn,
    lax,
    random
)
from math import prod
from typing import Tuple, Any, NamedTuple

class Hyperparams(NamedTuple):
    channel_axis: int
    epsilon: Any
    momentum: Any

def init(
    key,
    in_channels: int,
    channel_axis: int=0,
    epsilon=1e-5,
    momentum=0.9,
    mean_initializer=nn.initializers.zeros,
    var_initializer=nn.initializers.ones,
    dtype=None
) -> Tuple[None, Tuple[jax.Array, jax.Array], Hyperparams]:
    """Initialize parameters and hyperparameters for a batch norm layer.

    :param key: PRNG key for weight initialization.
    :param in_channels: Number of input feature dimensions/channels.
    :param eps: Small number added to variance to avoid divisions by zero.
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
    :returns hyperparams: NamedTuple containing the hyperparameters.
    """
    key1, key2 = random.split(key)
    moving_mean = mean_initializer(key1, (in_channels,), dtype)
    moving_var = var_initializer(key2, (in_channels,), dtype)

    return None, (moving_mean, moving_var), Hyperparams(
        channel_axis,
        epsilon,
        momentum
    )

def fwd(
    x: jax.Array,
    trainables: None,
    non_trainables: Tuple[jax.Array, jax.Array],
    hyperparams: Hyperparams,
    inference_mode: bool=False
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
    """Apply batch normalization without the learnable parameters.
 
    :param x: Input features to the batch norm. Must be of ``dtype``.
    :param trainables: Trainable weights for a batch norm. Should be None.
        Ignored.
    :param non_trainables: Non-trainable weights for a batch norm.
    :param hyperparams: NamedTuple containing the hyperparameters.
    :param inference_mode: Whether in inference or training mode. If in
        inference mode, the moving mean and variance are used to normalize input
        features. If in training mode, the batch mean and variance are used, and
        the moving mean and variance are updated. Default: False, training mode.
 
    :returns y: Batch normalized ``x``.
    :returns non_trainables: Updated non-trainables.

    .. note:
        If you wish to batch normalize using the moving mean and variance in
        inference mode, simply use ``mlax.nn.F`` and ``jax.nn.standarize``.

    .. note:
        The variance is calculated per ``jax.nn.standarize``'s implementation.
    """
    channel_axis = (
        hyperparams.channel_axis + len(x.shape) if hyperparams.channel_axis < 0
        else hyperparams.channel_axis + 1
    )
    broadcast_dims = (channel_axis,)

    if inference_mode is True:
        mean, variance = non_trainables
    else:
        reduce_dims = tuple(
            i for i in range(len(x.shape)) if i != channel_axis
        )

        n_elems = lax.convert_element_type(
            prod(tuple(d for i, d in enumerate(x.shape) if i != channel_axis)),
            x.dtype
        )

        mean = lax.div(
            lax.reduce(x, 0, lax.add, reduce_dims),
            n_elems
        )
        variance = lax.sub(
            lax.div(
                lax.reduce(
                    lax.integer_pow(x, 2), # integer_pow not in lax docs
                    0, lax.add, reduce_dims
                ),
                n_elems
            ),
            lax.integer_pow(mean, 2)
        )

        moving_mean, moving_var = non_trainables
        momentum = lax.convert_element_type(
            hyperparams.momentum, x.dtype
        )
        one_m_momentum = lax.convert_element_type(
            1.0 - hyperparams.momentum, x.dtype
        )
        moving_mean = lax.add(
            moving_mean * momentum,
            mean * one_m_momentum
        )
        moving_var = lax.add(
            moving_var * momentum,
            variance * one_m_momentum
        )
        non_trainables = (moving_mean, moving_var)
    
    return lax.mul(
        lax.sub(
            x,
            lax.broadcast_in_dim(mean, x.shape, broadcast_dims)
        ),
        lax.broadcast_in_dim(
            lax.rsqrt(
                lax.add(
                    variance,
                    lax.convert_element_type(
                        hyperparams.epsilon, x.dtype
                    )
                )
            ),
            x.shape, broadcast_dims 
        )
    ), non_trainables
