import jax
from jax import (
    nn,
    lax
)
from typing import Tuple, Any, NamedTuple, Sequence, Union

class Hyperparams(NamedTuple):
    broadcast_dims: Sequence[int]

def init(
    key: Any,
    in_feature_shape: Sequence[int],
    scaler_initializer=nn.initializers.ones,
    dtype=None
) -> Tuple[jax.Array, None, Hyperparams]:
    """Intialize parameters and hyperparametersfor a scaler layer.

    :param key: PRNG key for weight initialization.
    :param in_feature_shape: Shape of the input features to scale. Use ``None``
        on axes that do not require a scaler, use ``1`` on axes that require a
        scaler, and ``axis_length`` on axes that require a scaler for each of
        their elements.
    :param scaler_initializer: Initializer as defined by
        ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
        Default:: ones.
    :param dtype: Type of initialized scaler weight. Default: None.
        ``scaler_initializer``'s default.

    :returns trainables: Initialized scaler weight of shape
        ``tuple(in_feature_shape[dim] for dim in scaler_axis)``.
    :returns non_trainables: None.
    :returns hyperparams: NamedTuple containing the hyperparamters.
    """
    scaler_weight = scaler_initializer(
        key,
        tuple(axis for axis in in_feature_shape if axis is not None),
        dtype
    )

    return scaler_weight, None, Hyperparams(
        tuple(
            i + 1 for i, axis in enumerate(in_feature_shape) if axis is not None
        )
    )


def fwd(
    x: jax.Array,
    trainables: jax.Array,
    non_trainables: None,
    hyperparams: Hyperparams,
    inference_mode: bool=False
) -> jax.Array:
    """Scale input features.

    :param x: Input features to the scaler layer. Must be of ``dtype`` and of
        the shape as ``(n_batches, *in_feature_shape)``.
    :param trainables: Trainable weights for a scaler layer.
    :param non_trainables: Non-trainable weights for a scaler layer, should
        be None. Ignored.
    :param hyperparams: NamedTuple containing the hyperparameters. 
    :param inference_mode: Whether in inference or training mode. Ignored.
        Default: False.

    :returns y: Scaled ``x``.
    :returns non_trainables: None.
    """
    return lax.mul(
        x,
        lax.broadcast_in_dim(
            trainables, x.shape, hyperparams.broadcast_dims
        )
    ), None
