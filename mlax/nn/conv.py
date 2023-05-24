from jax import (
    Array,
    numpy as jnp,
    nn,
    lax,
    dtypes
)
from functools import reduce
from typing import Tuple, Sequence, Union, Optional, Hashable, Any
from mlax import Parameter, Module
from mlax._utils import (
    _canon_int_sequence,
    _canon_opt_int_sequence,
    _canon_padding,
    _canon_opt_dtype,
    _canon_precision_pair
)

class Conv(Module):
    """Convolution transformation layer."""
    def __init__(
        self,
        rng: Array,
        out_channels: int,
        filter_shape: Union[int, Sequence[int]],
        strides: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]]="VALID",
        input_dilation: Optional[Union[int, Sequence[int]]]=None,
        filter_dilation: Optional[Union[int, Sequence[int]]]=None,
        feature_group_count: int=1,
        batch_group_count: int=1,
        data_format: Union[str, Tuple[str, str, str]]="channel_last",
        precision=None,
        accum_dtype=None,
        kernel_initializer=nn.initializers.glorot_uniform(),
        dtype=jnp.float32
    ):
        """Initialize a Conv layer.

        :param rng: PRNG key.
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
        :param data_format: "channel_last", "channel_first", or a 3-tuple of
            strings as described in ``jax.lax.conv_general_dilated`` but without
            the batch axis "N".
        :param precision: See the ``precision`` parameter of
            `jax.lax.conv_general_dilated`_. Default: None.
        :param accum_dtype: See the ``preferred_element_type`` parameter of
            `jax.lax.conv_general_dilated`_. Default: None.
        :param kernel_initializer: Initializer for kernel of format "N..IO" as
            defined by ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
            Default:: glorot uniform.
        :param dtype: Type of initialized parameters. Default: float32.
        """
        super().__init__()

        self.rng = rng
        self.out_channels = int(out_channels)
        self.filter_shape = _canon_int_sequence(filter_shape)
        self.strides = _canon_int_sequence(strides)
        self.padding = _canon_padding(padding)
        self.input_dilation = _canon_opt_int_sequence(input_dilation)
        self.filter_dilation = _canon_opt_int_sequence(filter_dilation)
        self.feature_group_count = int(feature_group_count)
        self.batch_group_count = int(batch_group_count)
        self.data_format = (
            str(data_format) if isinstance(data_format, str)
            else tuple(str(s) for s in data_format[:3])
        )
        self.precision = _canon_precision_pair(precision)
        self.accum_dtype = _canon_opt_dtype(accum_dtype)
        self.kernel_initializer = kernel_initializer
        self.dtype = dtypes.canonicalize_dtype(dtype)

        self.conv_kernel = Parameter(trainable=True)
        self.dimension_numbers = None

    def init(self, x: Array) -> None:
        n_spatial_dims = x.ndim - 1
        filter_shape = _canon_int_sequence(self.filter_shape, n_spatial_dims)
        if isinstance(self.data_format, tuple):
            i_spec, kernel_spec, o_spec = self.data_format
            dims_map = {}
            dim = 0
            for c in i_spec:
                if c == "C":
                    channel_dim = dim
                else:
                    dims_map[c] = dim
                    dim += 1

            self.conv_kernel.data = self.kernel_initializer(
                self.rng,
                [*filter_shape, x.shape[channel_dim], self.out_channels],
                self.dtype
            )
            self.conv_kernel.data = lax.transpose(
                self.conv_kernel.data,
                [
                    n_spatial_dims + 1 if c == "O" else
                    n_spatial_dims if c == "I" else
                    dims_map[c] for c in self.data_format[1]
                ]
            )
        else:
            chars = reduce(
                lambda a, b: a + chr(97 + b),
                range(n_spatial_dims),
                ""
            ) # ab...

            if self.data_format == "channel_last":
                self.conv_kernel.data = self.kernel_initializer(
                    self.rng,
                    [*filter_shape, x.shape[-1], self.out_channels],
                    self.dtype
                )
                self.conv_kernel.data = lax.transpose(
                    self.conv_kernel.data,
                    [
                        n_spatial_dims + 1,
                        *range(n_spatial_dims),
                        n_spatial_dims
                    ]
                )
                i_spec = chars + "C" # ab...C
                kernel_spec = "O" + chars + "I" # Oab...I
                o_spec = i_spec
            elif self.data_format == "channel_first":
                self.conv_kernel.data = self.kernel_initializer(
                    self.rng,
                    [*filter_shape, x.shape[0], self.out_channels],
                    self.dtype
                )
                self.conv_kernel.data = lax.transpose(
                    self.conv_kernel.data,
                    [
                        n_spatial_dims + 1,
                        n_spatial_dims,
                        *range(n_spatial_dims)
                    ]
                )
                i_spec = "C" + chars # Cab...
                kernel_spec = "OI" + chars # OIab...
                o_spec = i_spec

        i_spec = "N" + i_spec
        o_spec = "N" + o_spec
        self.dimension_numbers = lax.conv_dimension_numbers(
            lax.broadcast(x, (1,)).shape, self.conv_kernel.data.shape,
            (i_spec, kernel_spec, o_spec)
        )

    def apply(
        self,
        x: Array,
        rng: None=None,
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> Tuple[Array, Any]:
        """Apply convolutions on input features."""
        n_spatial_dims = x.ndim - 1
        x = lax.broadcast(x, (1,))
        x = lax.conv_general_dilated(
            x,
            lax.convert_element_type(self.conv_kernel.data, x.dtype),
            _canon_int_sequence(self.strides, n_spatial_dims),
            _canon_padding(self.padding, n_spatial_dims),
            _canon_opt_int_sequence(self.input_dilation, n_spatial_dims),
            _canon_opt_int_sequence(self.filter_dilation, n_spatial_dims),
            self.dimension_numbers,
            self.feature_group_count,
            self.batch_group_count,
            self.precision,
            self.accum_dtype
        )
        return lax.squeeze(x, (0,))
