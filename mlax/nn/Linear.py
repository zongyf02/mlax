import jax
from jax import (
    nn,
    lax
)
from typing import Tuple, Sequence, Any
from mlax._utils import (
    _canon_opt_dtype,
    _canon_precision,
    _nn_hyperparams
)

@_nn_hyperparams
class LinearHp:
    transposed_kernel: bool
    precision: Any
    accum_dtype: Any

def init(
    key: Any,
    in_features: int,
    out_features: int,
    precision=None,
    accum_dtype=None,
    transposed_kernel=False,
    kernel_initializer=nn.initializers.glorot_uniform(),
    dtype=None
) -> Tuple[jax.Array, None, LinearHp]:
    """Intialize parameters and hyperparameters for a linear layer.

    :param key: PRNG key for weight initialization.
    :param in_features: Number of input features.
    :param out_features: Number of output features.
    :param precision: See ``precision`` parameter of
        ``jax.lax.dot <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot_general.html>``,
        which is used internally in the forward pass. Default: None.
    :param accum_dtype: See ``preferred_element_type`` parameter of
        ``jax.lax.dot <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot_general.html>``,
        which is used internally in the forward pass. Default: None.
    :param transposed_kernel: Whether the kernel is of the shape
        ``(out_features, in_features)`` or ``(in_features, out_features)``.
        Default: False, ``(in_features, out_features)``.
    :param kernel_initializer: Kernel initializer as defined by
        ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
        Default:: glorot uniform.
    :param dtype: Type of initialized kernel weight. Default: None.
        ``kernel_initializer``'s default.

    :returns trainables: Initialized kernel weight.
    :returns non_trainables: None.
    :returns hyperparams: LinearHp instance.
    """
    kernel_weight = kernel_initializer(
        key,
        (in_features, out_features),
        dtype 
    )
    if transposed_kernel:
        kernel_weight = lax.transpose(kernel_weight, (1, 0))

    hyperparams = LinearHp(
        transposed_kernel,
        _canon_precision(precision),
        _canon_opt_dtype(accum_dtype)
    )

    return kernel_weight, None, hyperparams

def fwd(
    x: jax.Array,
    trainables: jax.Array,
    non_trainables: None,
    hyperparams: LinearHp,
    inference_mode: bool=False
) -> jax.Array:
    """Apply linear transformation without bias to input features.

    :param x: Input features to the linear layer. Must be of ``dtype`` and of
        the shape ``(batch, in_features)``.
    :param trainables: Trainable weights for a linear layer.
    :param non_trainables: Non-trainable weights for a linear layer, should
        be None. Ignored.
    :param hyperparams: LinearHp instance.
    :param inference_mode: Whether in inference or training mode. Ignored.
        Default: False.

    :returns y: ``x`` with linear transformation applied. Shape
        ``(batch, out_features)``.
    :returns non_trainables: None.
    """
    contracting_dims = (1,) if hyperparams.transposed_kernel else (0,)
    return lax.dot_general(
        x,
        lax.convert_element_type(trainables, x.dtype),
        (((1,), contracting_dims), ((), ())),
        hyperparams.precision,
        hyperparams.accum_dtype
    ), None
