import jax
from jax import (
    nn,
    lax
)
from typing import Tuple, Any, NamedTuple, Sequence
from mlax.nn import _utils

class Hyperparams(NamedTuple):
    broadcast_dims: Sequence[int]
    dtype: Any

def init(
    key: Any,
    in_feature_shape: Sequence[int],
    scaler_dims: Sequence[int] = (0,),
    dtype=None,
    scaler_initializer=nn.initializers.ones,
    param_dtype=jax.numpy.float32
) -> Tuple[jax.Array, None, Hyperparams]:
    """Intialize parameters and hyperparametersfor a scaler layer.

    :param key: PRNG key for weight initialization.
    :param in_feature_shape: Shape of the input features.
    :param scaler_dims: Sequence indicating which dimensions in the input
        features are scaled. Default: (0,).
    :param dtype: Type of computation. Default: None, inferred from
        ``param_dtype``.
    :param scaler_initializer: Initializer as defined by
        ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
        Default:: ones.
    :param param_dtype: Type of initialized bias weight. Default: float32.

    :returns trainables: Initialized scaler weight of shape
        ``tuple(in_feature_shape[dim] for dim in bias_dims)``.
    :returns non_trainables: None.
    :returns hyperparams: NamedTuple containing the hyperparamters.
    """
    scaler_weight = scaler_initializer(
        key,
        tuple(in_feature_shape[dim] for dim in scaler_dims),
        param_dtype
    )

    return scaler_weight, None, Hyperparams(
        tuple(dim + 1 for dim in scaler_dims),
        _utils._canon_dtype(dtype, param_dtype)
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
            lax.convert_element_type(trainables, hyperparams.dtype), 
            x.shape, hyperparams.broadcast_dims
        )
    ), None
