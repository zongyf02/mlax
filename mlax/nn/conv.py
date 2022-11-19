from jax import (
    nn,
    lax
)
from functools import reduce

def init(
    key,
    in_channels,
    out_channels,
    filter_shape,
    kernel_initializer=nn.initializers.glorot_uniform(),
    dtype=None
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
    :param dtype: Type of initialized weights. Default: None, which is the
        ``kernel_initializer``'s default. 

    .. warning::
        ``filter_shape`` is used to infer the spatial dimension. Therefore, to
        specify a 2D filter of shape (3, 3), write ``filter_shape = (3, 3)`` and
        not ``filter_shape = 3``.

    :returns weight: Initialized kernel weight of shape
        ``filter_shape + (in_channels, out_channels)``.
    """
    kernel_shape = None
    kernel_shape = filter_shape + (in_channels, out_channels)
  
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
    feature_group_count=1,
    batch_group_count=1,
    channel_first=False,
    precision=None,
    preferred_element_type=None
):
    """Applies convolutions on input features.

    :param x: Input features to the convolutional transform. Must be compatible
        with the ``lhr_spec`` of ``dimension_numbers``.
    :param weights: Initialized kernel weights for a convolutional transform.
        Must be compatible with the ``rhs_spec`` of ``dimension_numbers``.
    :param strides: See the ``strides`` parameter of
        `jax.lax.conv_general_dilated`_.
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
    :param feature_group_count: See the ``feature_group_count`` parameter of
        `jax.lax.conv_general_dilated`_. Can be used to perform seperable
        convolutions. Default: 1.
    :param batch_group_count: See the ``batch_group_count`` parameter of
        `jax.lax.conv_general_dilated`_. Default: 1.
    :param channel_first: Whether features are channel-first or channel-last.
        Default: False, channel-last.
    :param precision: See the ``precision`` parameter of
        `jax.lax.conv_general_dilated`_. Default: None.
    :param preferred_element_type: See the ``preferred_element_type`` parameter
        of `jax.lax.conv_general_dilated`_. Default: None.

    .. _jax.lax.conv_general_dilated:
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html#jax.lax.conv_general_dilated
    
    :returns y: Convolution on input features compatible with the ``out_spec``
        of ``dimension_numbers``.
    """
    dims = len(x.shape) - 2
    feature_spec = reduce(lambda s, i: s + str(i), range(dims), "")
    if channel_first:
        lhs_spec = "NC" + feature_spec
        rhs_spec = feature_spec + "IO"
    else:
        lhs_spec = "N" + feature_spec + "C"
        rhs_spec = feature_spec + "IO"
    dimension_numbers = (lhs_spec, rhs_spec, lhs_spec)
    
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
