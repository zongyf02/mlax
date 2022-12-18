import jax
from jax import (
    nn,
    lax
)
from typing import Tuple, Any, NamedTuple, Sequence, Optional

class Hyperparams(NamedTuple):
    broadcast_dims: Sequence[int]

def init(
    key: Any,
    in_feature_shape: Sequence[Optional[int]],
    bias_initializer=nn.initializers.zeros,
    dtype=None
) -> Tuple[jax.Array, None, Hyperparams]:
    """Intialize parameters and hyperparameters for a bias layer.

    :param key: PRNG key for weight initialization.
    :param in_feature_shape: Shape of the input features to add bias to. Use
        ``None`` on axes that do not require a bias, use ``1`` on axes that
        require a single bias term, and ``axis_length`` on axes that require a
        bias term for each of their elements.
    :param bias_initializer: Initializer as defined by
        ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
        Default:: zeros.
    :param dtype: Type of initialized bias weight. Default: None.
        ``bias_initializer``'s default.

    :returns trainables: Initialized bias weight of shape
        ``tuple(in_feature_shape[dim] for dim in bias_axis)``.
    :returns non_trainables: None.
    :returns hyperparams: NamedTuple containing the hyperparamters.
    """
    bias_weight = bias_initializer(
        key,
        tuple(axis for axis in in_feature_shape if axis is not None),
        dtype
    )

    return bias_weight, None, Hyperparams(
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
    """Add bias to input features.

    :param x: Input features to the bias layer. Must be of ``dtype`` and of the
        shape ``(n_batches, *in_feature_shape)``.
    :param trainables: Trainable weights for a bias layer.
    :param non_trainables: Non-trainable weights for a bias layer, should
        be None. Ignored.
    :param hyperparams: NamedTuple containing the hyperparameters. 
    :param inference_mode: Whether in inference or training mode. Ignored.
        Default: False.

    :returns y: ``x`` plus bias.
    :returns non_trainables: None.
    """
    return lax.add(
        x,
        lax.broadcast_in_dim(
            lax.convert_element_type(trainables, x.dtype),
            x.shape,
            hyperparams.broadcast_dims
        )
    ), None
