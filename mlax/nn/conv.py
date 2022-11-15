from jax import (
    nn,
    lax
)

def init(
    key,
    in_channels,
    out_channels,
    filter_shape,
    kernel_spec=None,
    kernel_initializer=nn.initializers.glorot_uniform(in_axis=1, out_axis=0),
    dtype="float32"
):
    """Intialize weights for a convolutional transform.

    :param key: PRNG key for weight initialization.
    :param in_channels: Number of input feature dimensions/channels.
    :param out_channels: Number of desired output feature dimensions/channels.
    :param filter_shape: Shape of filter used on input features. Used to infer
        the spatial dimension.
    :param kernel_initializer: Initializer as defined by
        ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
        Default:: glorot uniform with the input axis at 1 and output axis at 0.
    :param kernel_spec: String specifying the kerel spec following the
        format in xla_client.py ('I' for input feature dimension, 'O' for output
        feature dimension, any other character for spatial dimensions.)
        Default: None, output feature dimension is first, then input feature
        dimension, then spatial dimensions.
    :param dtype: Type of initialized weights. Default: float32.

    .. note::
        If you change the ``kernel_spec``, for example, to "HWIO" such that the
        input and output dimensions are last, make sure to also update the
        initializer, in this case, the glorot uniform initializer to
        ``in_axis = -2`` and ``out_axis = -1`` to match the change in kernel
        specification.

    .. warning::
        ``filter_shape`` is used to infer the spatial dimension. Therefore, to
        specify a 2D filter of shape (3, 3), write ``filter_shape = (3, 3)`` and
        not ``filter_shape = 3``.

    :returns weight: Initialized kernel weight of shape
        (out_channels, in_channels, ``filter_shape``).
    """
    kernel_shape = None
    if kernel_spec is None:
        kernel_shape = (out_channels, in_channels) + filter_shape
    else:
        filter_shape_iter = iter(filter_shape)
        kernel_shape = [out_channels if c == 'O' else
            in_channels if c == 'I' else
            next(filter_shape_iter) for c in kernel_spec
        ]
    
    return kernel_initializer(
        key,
        kernel_shape,
        dtype
    )

def fwd(
    x,
    weights,
    strides,
    padding="VALID",
    input_dilation=None,
    filter_dilation=None,
    dimension_numbers=None,
    feature_group_count=1,
    batch_group_count=1,
    precision=None,
    preferred_element_type=None
):
    """Applies convolutions on input features.

    :param x: Input features to the convolutional transform. Must be compatible
        with the ``lhr_spec`` of ``dimension_numbers``.
    :param weights: Initialized kernel weights for a convolutional transform.
        Must be compatible with the ``rhs_spec`` of ``dimension_numbers``.
    :param padding: See the ``padding`` parameter of
        `jax.lax.conv_general_dilated`_.
    :param input_dilation: Dilation rate applied to the input features, also
        known as transposed convolution dilation rate. See the ``lhs_dilation``
        parameter of `jax.lax.conv_general_dilated`_. Default: None, no input
        dilation.
    :param filter_dilation: Dilation rate applied to each filter, also known as
        atrous convolution dilation rate. See the ``rhs_dilation``
        parameter of `jax.lax.conv_general_dilated`_. Default: None, no filter
        dilation.
    :param dimension_numbers: See the ``dimension_numbers`` parameter of
        `jax.lax.conv_general_dilated`_. Default: None, equivalent to
        ``("NC...", "OI...", "NC...")``. Inputs and outputs are batched and
        channel-first, kernel has output feature dimension and input feature
        dimension first.
    :param feature_group_count: See the ``feature_group_count`` parameter of
        `jax.lax.conv_general_dilated`_. Can be used to perform seperable
        convolutions. Default: 1.
    :param batch_group_count: See the ``batch_group_count`` parameter of
        `jax.lax.conv_general_dilated`_. Default: 1.
    :param precision: See the ``precision`` parameter of
        `jax.lax.conv_general_dilated`_. Default: None.
    :param preferred_element_type: See the ``preferred_element_type`` parameter
        of `jax.lax.conv_general_dilated`_. Default: None.

    .. _jax.lax.conv_general_dilated:
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html#jax.lax.conv_general_dilated
    
    :returns y: Convolution on input features compatible with the ``out_spec``
        of ``dimension_numbers``.
    """
    return lax.conv_general_dilated(
        x,
        weights,
        strides,
        padding,
        input_dilation,
        filter_dilation,
        dimension_numbers,
        feature_group_count,
        batch_group_count,
        precision,
        preferred_element_type
    )
