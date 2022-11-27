import jax
from jax import (
    nn,
    lax
)
from typing import Tuple, Sequence, Any, NamedTuple

class Hyperparams(NamedTuple):
    precision: Any
    accum_dtype: Any

def init(
    key: Any,
    in_feature_shape: Sequence[int],
    out_feature_shape: Sequence[int],
    precision=None,
    accum_dtype=None,
    kernel_initializer=nn.initializers.glorot_uniform(in_axis=-1, out_axis=0),
    dtype=None
) -> Tuple[jax.Array, None, Hyperparams]:
    """Intialize variables for a linear transform.

    :param key: PRNG key for weight initialization.
    :param in_feature_shape: Shape of input features.
    :param out_feature_shape: Shape of output features.
    :param precision: See ``precision`` parameter of
        ``jax.lax.dot <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot_general.html>``,
        which is used internally in the forward pass. Default: None.
    :param accum_dtype: See ``preferred_element_type`` parameter of
        ``jax.lax.dot <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot_general.html>``,
        which is used internally in the forward pass. Default: None.
    :param kernel_initializer: Kernel initializer as defined by
        ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
        Default:: glorot uniform.
    :param dtype: Type of initialized kernel weight. Default: None, which means
        the ``kernel_initializer``'s default.

    :returns trainables: Initialized kernel weight of shape
        ``(*out_feature_shape, *in_feature_shape)``.
    :returns non_trainables: None.
    :returns hyperparams: NamedTuple containing the hyperparameters.
    """
    kernel_weight = kernel_initializer(
        key,
        (*out_feature_shape, *in_feature_shape),
        dtype 
    )
    hyperparams = Hyperparams(
        precision,
        accum_dtype
    )

    return kernel_weight, None, hyperparams

def fwd(
    x: jax.Array,
    trainables: jax.Array,
    non_trainables: None,
    hyperparams: Hyperparams,
    inference_mode: bool=False
) -> jax.Array:
    """Apply linear transformation without bias to input features.

    :param x: Input features to the linear transform. Must be of the shape
        ``(n_batches, *in_feature_shape)``.
    :param trainables: Trainable weights for a linear transform.
    :param non_trainables: Non-trainable weights for a linear transform, should
        be None. Ignored.
    :param hyperparams: NamedTuple containing the hyperparameters.
    :param inference_mode: Whether in inference or training mode. Ignored.
        Default: False.

    :returns y: ``x`` with linear transformation applied. Shape
        ``(n_batches, *out_feature_shape)``.
    :returns non_trainables: Unchanged ``non_trainables``.
    """
    input_dim = len(x.shape)
    kernel_dim = len(trainables.shape)
    input_contracting_dims = tuple(range(1, input_dim))
    kernel_contracting_dims = tuple(range(kernel_dim-input_dim+1, kernel_dim))
    return lax.dot_general(
        x,
        trainables,
        ((input_contracting_dims, kernel_contracting_dims), ((), ())),
        hyperparams.precision,
        hyperparams.accum_dtype
    ), non_trainables
