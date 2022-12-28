import jax
from jax import (
    nn,
    lax
)
from functools import reduce
from operator import add
from typing import Tuple, Any, Sequence, Union, Optional
from mlax._utils import (
    _canon_int_sequence,
    _canon_opt_int_sequence,
    _canon_padding,
    _canon_opt_dtype,
    _canon_precision,
    _nn_hyperparams
)

@_nn_hyperparams
class ConvHp:
    window_strides: Sequence[int] 
    padding: Union[str, Sequence[Tuple[int, int]]] 
    input_dilation: Optional[Sequence[int]]
    filter_dilation: Optional[Sequence[int]]
    feature_group_count: int
    batch_group_count: int
    dimension_numbers: Any
    precision: Any 
    accum_dtype: Any

def init(
    key: Any,
    ndims: int,
    in_channels: int,
    out_channels: int,
    filter_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]]=1,
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "VALID",
    input_dilation: Optional[Union[int, Sequence[int]]] = None,
    filter_dilation: Optional[Union[int, Sequence[int]]] = None,
    feature_group_count = 1,
    batch_group_count = 1,
    channel_last: bool=False,
    precision=None,
    accum_dtype=None,
    kernel_initializer=nn.initializers.glorot_uniform(),
    dtype=None
) -> Tuple[jax.Array, None, ConvHp]:
    """Intialize parameters and hyperparameters for a convolutional layer.

    :param key: PRNG key for weight initialization.
    :param ndims: Number of input spatial dimensions.
    :param in_channels: Number of input feature dimensions/channels.
    :param out_channels: Number of desired output feature dimensions/channels.
    :param filter_shape: An integer or a sequence of ``ndims`` integers,
        specifying the shape of the filters used on input features. A single
        integer specifies the same value for all spatial dimensions.
    :param strides: An integer or a sequence of ``ndims`` integers, specifying
        the strides of the convolution along the spatial dimensions. A single
        integer specifies the same value for all spatial dimensions. Default: 1.
    :param padding: String, integer, or a sequence of `ndims` integers or
        integer tuple pairs that give the padding to apply before and after
        each spatial dimension. If integer, the same padding is applied before
        and after all spatial dimensions. If a sequence of integers, then the
        same padding is applied before and after each spatial dimension.
        See the ``padding`` parameter of `jax.lax.conv_general_dilated`_.
    :param input_dilation: None, an integer, or a sequence of ``ndims``
        integers, specifying the transposed convolution dilation rate in each
        spatial dimension. See the ``lhs_dilation`` parameter of
        `jax.lax.conv_general_dilated`_. Default: None, no input dilation.
    :param filter_dilation: None, an integer, or a sequence of ``ndims``
        integers, specifying the atrous convolution dilation rate. See the
        ``rhs_dilation`` parameter of `jax.lax.conv_general_dilated`_. Default:
        None, no filter dilation.
    :param feature_group_count: See the ``feature_group_count`` parameter of
        `jax.lax.conv_general_dilated`_. Can be used to perform group and
        seperable convolutions. Default: 1.
    :param batch_group_count: See the ``batch_group_count`` parameter of
        `jax.lax.conv_general_dilated`_. Default: 1.
    :param channel_last: Whether features are channel-last or first. Default:
        False, channel-first.
    :param precision: See the ``precision`` parameter of
        `jax.lax.conv_general_dilated`_. Default: None.
    :param accum_dtype: See the ``preferred_element_type`` parameter of
        `jax.lax.conv_general_dilated`_. Default: None.
    :param kernel_initializer: Kernel initializer as defined by
        ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
        Default:: glorot uniform.
    :param dtype: Type of initialized kernel weight. Default: None.
        ``kernel_initializer``'s default.

    :returns trainables: Initialized kernel weight.
    :returns non_trainables: None.
    :returns hyperparams: ConvHp instance.

    .. note:
        By default, because ``kernel_in_axis=1`` and ``kernel_out_axis=0``, the
        kernel laoyout is ``OI...``.

    .. note:
        If you override either ``kernel_in_axis`` or ``kernel_out_axis``, also
        override the default ``kernel_initializer`` to have matching
        ``in_axis`` and ``out_axis``.

    .. _jax.lax.conv_general_dilated:
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html
    """
    filter_shape = _canon_int_sequence(filter_shape, ndims)
    kernel_weight = kernel_initializer(
        key,
        (*filter_shape, in_channels, out_channels),
        dtype
    )

    chars = tuple(str(i) for i in range(ndims))
    if channel_last:
        io_spec = reduce(add, chars, "N") + "C" # N...C
        kernel_spec = reduce(add, chars, "O") + "I" # O...I
        kernel_weight = lax.transpose(
            kernel_weight,
            (ndims+1, *range(ndims), ndims)
        )
    else:
        io_spec = reduce(add, chars, "NC") # NC...
        kernel_spec = reduce(add, chars, "OI") # OI...
        kernel_weight = lax.transpose(
            kernel_weight,
            (ndims+1, ndims, *range(ndims))
        )

    dummy_shape = (None,) * (ndims + 2)
    dimension_numbers = lax.conv_dimension_numbers(
        dummy_shape, dummy_shape,
        (io_spec, kernel_spec, io_spec)
    )

    hyperparams = ConvHp(
        _canon_int_sequence(strides, ndims),
        _canon_padding(padding, ndims),
        _canon_opt_int_sequence(input_dilation, ndims),
        _canon_opt_int_sequence(filter_dilation, ndims),
        feature_group_count,
        batch_group_count,
        dimension_numbers,
        _canon_precision(precision),
        _canon_opt_dtype(accum_dtype)
    )

    return kernel_weight, None, hyperparams

def fwd(
    x: jax.Array,
    trainables: jax.Array,
    non_trainables: None,
    hyperparams: ConvHp,
    inference_mode: bool=False
) -> jax.Array:
    """Applies convolutions on input features.

    :param x: Batched input features to the convolutional layer. Must be of
        ``dtype`` and compatible with ``channel_last``.
    :param trainables: Trainable weights for a convolutional layer.
    :param non_trainables: Non-trainable weights for a convolutional layer,
        should be None. Ignored.
    :param hyperparams: ConvHp instance.
    :param inference_mode: Whether in inference or training mode. Ignored.
        Default: False.
 
    :returns y: Convolution on ``x``.
    :returns non_trainables: None.
    """

    return lax.conv_general_dilated(
        x,
        lax.convert_element_type(trainables, x.dtype),
        hyperparams.window_strides,
        hyperparams.padding,
        hyperparams.input_dilation,
        hyperparams.filter_dilation,
        hyperparams.dimension_numbers,
        hyperparams.feature_group_count,
        hyperparams.batch_group_count,
        hyperparams.precision,
        hyperparams.accum_dtype
    ), None
