import jax
from jax import (
    nn,
    lax
)
from typing import Tuple, Any, NamedTuple, Sequence

class Hyperparams(NamedTuple):
    broadcast_dims: Sequence[int] 

def init(
    key: Any,
    in_feature_shape: Sequence[int],
    bias_dims: Sequence[int] = (0,),
    bias_initializer=nn.initializers.zeros,
    dtype=None
) -> Tuple[jax.Array, None, Hyperparams]:
    """Intialize variables for a bias transform.

    :param key: PRNG key for weight initialization.
    :param in_feature_shape: Shape of the input features.
    :param bias_dims: Sequence indicating to which dimensions in the input
        features to add bias. Default: (0,).
    :param bias_initializer: Initializer as defined by
        ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
        Default:: zeros.
    :param dtype: Type of initialized bias weight. Default: None, which means
        the ``bias_initializer``'s default.

    :returns trainables: Initialized bias weight of shape
        ``[in_feature_shape[dim] for dim in bias_dims]``.
    :returns non_trainables: None.
    :returns hyperparams: NamedTuple containing the hyperparamters.
    """
    bias_weight = bias_initializer(
        key,
        tuple(in_feature_shape[dim] for dim in bias_dims),
        dtype
    )

    return bias_weight, None, Hyperparams(tuple(dim + 1 for dim in bias_dims))

def fwd(
    x: jax.Array,
    trainables: jax.Array,
    non_trainables: None,
    hyperparams: Hyperparams,
    inference_mode: bool=False
) -> jax.Array:
    """Add bias to input features.

    :param x: Input features to the bias transform. Must be of the shape as
        ``(n_batches, *in_feature_shape)``.
    :param trainables: Trainable weights for a bias transform.
    :param non_trainables: Non-trainable weights for a bias transform, should
        be None. Ignored.
    :param hyperparams: NamedTuple containing the hyperparameters. 
    :param inference_mode: Whether in inference or training mode. Ignored.
        Default: False.

    :returns y: ``x`` plus bias.
    :returns non_trainables: Unchanged ``non_trainables``.
    """
    return lax.add(
        x,
        lax.broadcast_in_dim(trainables, x.shape, hyperparams.broadcast_dims)
    ), non_trainables
