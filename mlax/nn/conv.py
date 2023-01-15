from mlax import Parameter, Module
from jax import (
    numpy as jnp,
    nn,
    lax
)
from functools import reduce
from typing import Tuple, Any, Sequence, Union, Optional
from mlax._utils import (
    _canon_int_sequence,
    _canon_opt_int_sequence,
    _canon_padding,
    _canon_dtype,
    _canon_opt_dtype,
    _canon_precision_pair
)


class Conv(Module):
    """Convolution transformation layer."""
    def __init__(
        self,
        rng: Any,
        n_spatial_dims: int,
        out_channels: int,
        filter_shape: Union[int, Sequence[int]],
        strides: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "VALID",
        input_dilation: Optional[Union[int, Sequence[int]]] = None,
        filter_dilation: Optional[Union[int, Sequence[int]]] = None,
        feature_group_count: int = 1,
        batch_group_count: int = 1,
        channel_last: bool=False,
        precision=None,
        accum_dtype=None,
        kernel_initializer=nn.initializers.glorot_uniform(),
        dtype=jnp.float32
    ):
        """Initialize a Conv layer.

        :param rng: PRNG key for weight initialization.
        :param n_spatial_dims: Number of input spatial dimensions.
        :param out_channels: Number of desired output channels.
        :param filter_shape: An integer or a sequence of ``n_spatial_dims``
            integers, specifying the shape of the filters. A single integer
            specifies the same value for all spatial dimensions.
        :param strides: An integer or a sequence of ``n_spatial_dims`` integers,
            specifying the strides of the convolution along the spatial
            dimensions. A single integer specifies the same value for all
            spatial dimensions. Default: 1.
        :param padding: String, integer, or a sequence of ``n_spatial_dims``
            integers or integer tuple pairs that gives the padding to apply
            before and after each spatial dimension. If integer, the same
            padding is applied before and after all spatial dimensions. If a
            sequence of integers, the same padding is applied before and after
            each spatial dimension. See the ``padding`` parameter of
            `jax.lax.conv_general_dilated`_.
        :param input_dilation: None, an integer, or a sequence of
            ``n_spatial_dims`` integers, specifying the transposed convolution
            dilation rate in each spatial dimension. See the ``lhs_dilation``
            parameter of `jax.lax.conv_general_dilated`_. Default: None, no
            input dilation.
        :param filter_dilation: None, an integer, or a sequence of
            ``n_spatial_dims`` integers, specifying the atrous convolution
            dilation rate. See the ``rhs_dilation`` parameter of
            `jax.lax.conv_general_dilated`_. Default: None, no filter dilation.
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
        """
        super().__init__()
        self.initialized = False

        self._rng =  Parameter(trainable=False, data=rng)
        self._n_spatial_dims = int(n_spatial_dims)
        self._out_channels = int(out_channels)
        self._filter_shape = _canon_int_sequence(
            filter_shape, self._n_spatial_dims
        )
        self._kernel_initializer = kernel_initializer
        self._dtype = _canon_dtype(dtype)

        self.kernel_weight = Parameter(trainable=True)
        self.strides = _canon_int_sequence(strides, self._n_spatial_dims)
        self.padding = _canon_padding(padding, self._n_spatial_dims)
        self.input_dilation = _canon_opt_int_sequence(
            input_dilation, self._n_spatial_dims
        )
        self.filter_dilation = _canon_opt_int_sequence(
            filter_dilation, self._n_spatial_dims
        )
        self.feature_group_count = int(feature_group_count)
        self.batch_group_count = int(batch_group_count)
        self.channel_last = bool(channel_last)
        self.precision = _canon_precision_pair(precision)
        self.accum_dtype = _canon_opt_dtype(accum_dtype)

        chars = reduce(
            lambda a, b: a + chr(97 + b),
            range(self._n_spatial_dims),
            ""
        ) # ab...
        if self.channel_last:
            io_spec = "N" + chars + "C" # Nab...C
            kernel_spec = "O" + chars + "I" # Oab...I
        else:
            io_spec = "NC" + chars # NCab...
            kernel_spec = "OI" + chars # OIab...
        dummy_shape = [None] * (self._n_spatial_dims + 2)
        self.dimension_numbers = lax.conv_dimension_numbers(
            dummy_shape, dummy_shape,
            (io_spec, kernel_spec, io_spec)
        )
        
    def _build(self, x):
        """Initialize an uninitialized conv layer."""
        if self.channel_last:
            kernel_weight = self._kernel_initializer(
                self._rng.data,
                (*self._filter_shape, x.shape[-1], self._out_channels),
                self._dtype
            )
            self.kernel_weight.data = lax.transpose(
                kernel_weight,
                (
                    self._n_spatial_dims + 1,
                    *range(self._n_spatial_dims),
                    self._n_spatial_dims
                )
            )
        else:
            kernel_weight = self._kernel_initializer(
                self._rng.data,
                (*self._filter_shape, x.shape[0], self._out_channels),
                self._dtype
            )
            self.kernel_weight.data = lax.transpose(
                kernel_weight,
                (
                    self._n_spatial_dims + 1,
                    self._n_spatial_dims,
                    *range(self._n_spatial_dims)
                )
            )

        del self._rng
        del self._n_spatial_dims
        del self._out_channels
        del self._filter_shape
        del self._kernel_initializer
        del self._dtype

        self.initialized = True

    
    def __call__(self, x, rng=None, inference_mode=False):
        """Apply convolutions on input features.

        :param self: Conv layer.
        :param x: Input features to the Conv layer. Must be unbatched and
        thus having ``n_spatial_dims + 1`` dimensions and be compatible with
            ``channel_last``.
        :param rng: PRNG key. Ignored. Default: None.
        :param inference_mode: Whether in inference or training mode. Ignored.
            Default: False.
        
        :returns: Convolution on ``x``.
        :returns: Conv layer with updated state. Possibly the same object as
            ``self``.
        """
        if not self.initialized:
            self._build(x)

        x = lax.broadcast(x, (1,))
        x = lax.conv_general_dilated(
            x,
            lax.convert_element_type(self.kernel_weight.data, x.dtype),
            self.strides,
            self.padding,
            self.input_dilation,
            self.filter_dilation,
            self.dimension_numbers,
            self.feature_group_count,
            self.batch_group_count,
            self.precision,
            self.accum_dtype
        )
        return lax.squeeze(x, (0,)), self
