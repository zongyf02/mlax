from jax import (
    Array,
    numpy as jnp,
    random,
    lax
)
from mlax._utils import (
    _canon_int_sequence,
    _canon_opt_int_sequence,
    _canon_padding,
    _compute_std_stats,
    _standadize
)
from math import prod, sqrt
from typing import Any, Tuple, Sequence, Union, Callable, Optional, Hashable

def identity(x: Any) -> Any:
    """Identity function.
    
    :param x: Input features.

    :returns y: ``x``.
    """
    return x

def dropout(
    x: Array,
    rng: Any,
    rate: float,
    axis: Union[int, Sequence[int]]
) -> Array:
    """Apply random dropouts to input features.

    :param x: Input features.
    :param rng: PRNG key for randomizing dropouts.
    :param rate: Probability at which each element is droped out. Must be in
        [0, 1).
    :param axis: Axis or sequence of axes to drop features along.

    :returns y: ``x`` with dropouts applied.
    """
    axis = _canon_int_sequence(axis, 1)
    prob = 1.0 - rate
    mask = lax.broadcast_in_dim(
        random.bernoulli(rng, prob, [x.shape[i] for i in axis]), x.shape, axis
    )
    return lax.select(
        mask,
        lax.div(x, lax.convert_element_type(prob, x.dtype)),
        lax.full_like(x, 0)
    )

def pool(
    x: Array,
    init_value: Any,
    reduce_fn: Callable[[Any, Any], Any],
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]] = 1,
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "VALID",
    input_dilation: Optional[Union[int, Sequence[int]]] = None,
    window_dilation: Optional[Union[int, Sequence[int]]] = None,
    data_format: str="channel_last"
) -> Array:
    """Apply an arbitrary reduce function over poolings windows of input
        features.

    :param x: Input features. Must have be unbatched thus having
        ``n_spatial_dims + 1`` dimensions.
    :param init_value: Initial value of the reduce function over each pooling
        window.
    :param reduce_fn: Reduce function.
    :param window_shape: An integer or a sequence of ``n_spatial_dims``
        integers, specifying the shape of the pooling window used on input
        features. A single integer specifies the same value for all spatial
        dimensions.
    :param strides: An integer or a sequence of ``n_spatial_dims`` integers,
        specifying the strides of the window shape along the spatial dimensions.
        A single integer specifies the same value for all spatial dimensions.
        Default: 1.
    :param padding: String, integer, or a sequence of `n_spatial_dims` integers
        or integer tuple pairs that give the padding to apply before and after
        each spatial dimension. If integer, the same padding is applied before
        and after all spatial dimensions. If a sequence of integers, then the
        same padding is applied before and after each spatial dimension.
        See the ``padding`` parameter of `jax.lax.reduce_window`_, which is used
        internally.
    :param input_dilation: None, an integer, or a sequence of ``n_spatial_dims``
        integers, specifying the input dilation rate in each spatial dimension.
        See the ``base_dilation`` parameter of `jax.lax.reduce_window`_.
        Default: None, no input dilation.
    :param window_dilation: None, an integer, or a sequence of ``n_spatial_dims``
        integers, specifying the window dilation rate in each spatial dimension.
        See the ``window_dilation`` parameter of `jax.lax.reduce_window`_.
        Default: None, no window dilation.
    :param data_format: "channel_last", "channel_first", or a string
        representing the kernel spec as described in
        ``jax.lax.conv_general_dilated``, but  without `N` the batch dimension.
        Default: "channel_last".

    :returns y: ``x`` with pooling applied.

    .. _jax.lax.reduce_window:
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.reduce_window.html
    """
    n_spatial_dims = x.ndim - 1
    window_shape = _canon_int_sequence(window_shape, n_spatial_dims)
    strides = _canon_int_sequence(strides, n_spatial_dims)
    padding = _canon_padding(padding, n_spatial_dims)
    input_dilation = _canon_opt_int_sequence(input_dilation, n_spatial_dims)
    window_dilation = _canon_opt_int_sequence(window_dilation, n_spatial_dims)

    if data_format == "channel_last":
        window_shape = window_shape + (1,)
        strides = strides + (1,)
        padding = padding + ((0, 0),) if isinstance(padding, tuple) else padding
        input_dilation = (
            input_dilation + (1,) if isinstance(input_dilation, tuple)
            else input_dilation
        )
        window_dilation = (
            window_dilation + (1,) if isinstance(window_dilation, tuple)
            else window_dilation
        )
    elif data_format == "channel_first":
        window_shape = (1,) + window_shape
        strides = (1,) + strides
        padding = ((0, 0),) + padding if isinstance(padding, tuple) else padding
        input_dilation = (
            (1,) + input_dilation if isinstance(input_dilation, tuple) else
            input_dilation
        )
        window_dilation = (
            (1,) + window_dilation if isinstance(window_dilation, tuple) else
            window_dilation
        )
    else:
        channel_dim = data_format.index("C")
        window_shape = (
            window_shape[:channel_dim] + (1,) + window_shape[channel_dim:]
        )
        strides = (
            strides[:channel_dim] + (1,) + strides[channel_dim:]
        )
        padding = (
            padding[:channel_dim] + ((0, 0),) + padding[channel_dim:]
            if isinstance(padding, tuple) else padding
        )
        input_dilation = (
            input_dilation[:channel_dim] + (1,) + input_dilation[channel_dim:]
            if isinstance(input_dilation, tuple) else input_dilation
        )
        window_dilation = (
            window_dilation[:channel_dim] + (1,) + window_dilation[channel_dim:]
            if isinstance(window_dilation, tuple) else window_dilation
        )

    return lax.reduce_window(
        x,
        init_value,
        reduce_fn,
        window_shape,
        strides,
        padding,
        input_dilation,
        window_dilation
    )

def max_pool(
    x: Array,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]] = 1,
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "VALID",
    input_dilation: Optional[Union[int, Sequence[int]]] = None,
    window_dilation: Optional[Union[int, Sequence[int]]] = None,
    data_format: str="channel_last"
) -> Array:
    """Apply max pooling over input features.

    :param x: Input features. Must have be unbatched thus having
        ``n_spatial_dims + 1`` dimensions.
    :param window_shape: See the ``window_shape`` parameter of ``pooling``.
    :param strides: See the ``strides`` parameter of ``pooling``. Default: 1.
    :param padding: See the ``padding`` parameter of ``pooling``.
    :param input_dilation: See the ``input_dilation`` parameter of ``pooling``.
        Default: None, no input dilation.
    :param window_dilation: See the ``window_dilation`` parameter of
        ``pooling``. Default: None, no window dilation.
    :param data_format: "channel_last", "channel_first", or a string
        representing the kernel spec as described in
        ``jax.lax.conv_general_dilated``, but without `N` the batch dimension.
        Default: "channel_last".
    
    :returns y: ``x`` with max pooling applied.
    """
    return pool(
        x,
        -jnp.inf,
        lax.max,
        window_shape,
        strides,
        padding,
        input_dilation,
        window_dilation,
        data_format
    )

def sum_pool(
    x: Array,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]] = 1,
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "VALID",
    input_dilation: Optional[Union[int, Sequence[int]]] = None,
    window_dilation: Optional[Union[int, Sequence[int]]] = None,
    data_format: str="channel_last"
) -> Array:
    """Apply sum pooling over input features.

    :param x: Input features. Must have be unbatched thus having
        ``n_spatial_dims + 1`` dimensions.
    :param window_shape: See the ``window_shape`` parameter of ``pooling``.
    :param strides: See the ``strides`` parameter of ``pooling``. Default: 1.
    :param padding: See the ``padding`` parameter of ``pooling``.
    :param input_dilation: See the ``input_dilation`` parameter of ``pooling``.
        Default: None, no input dilation.
    :param window_dilation: See the ``window_dilation`` parameter of
        ``pooling``. Default: None, no window dilation.
    :param data_format: "channel_last", "channel_first", or a string
        representing the kernel spec as described in
        ``jax.lax.conv_general_dilated``, but without `N` the batch dimension.
        Default: "channel_last".
    
    :returns y: ``x`` with sum pooling applied.
    """
    return pool(
        x,
        0,
        lax.add,
        window_shape,
        strides,
        padding,
        input_dilation,
        window_dilation,
        data_format
    )

def avg_pool(
    x: Array,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]] = 1,
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "VALID",
    input_dilation: Optional[Union[int, Sequence[int]]] = None,
    window_dilation: Optional[Union[int, Sequence[int]]] = None,
    data_format: str="channel_last"
) -> Array:
    """Apply average pooling over input features.

    :param x: Input features. Must have be unbatched thus having
        ``n_spatial_dims + 1`` dimensions.
    :param window_shape: See the ``window_shape`` parameter of ``pooling``.
    :param strides: See the ``strides`` parameter of ``pooling``. Default: 1.
    :param padding: See the ``padding`` parameter of ``pooling``.
    :param input_dilation: See the ``input_dilation`` parameter of ``pooling``.
        Default: None, no input dilation.
    :param window_dilation: See the ``window_dilation`` parameter of
        ``pooling``. Default: None, no window dilation.
    :param data_format: "channel_last", "channel_first", or a string
        representing the kernel spec as described in
        ``jax.lax.conv_general_dilated``, but without `N` the batch dimension.
        Default: "channel_last".

    :returns y: ``x`` with average pooling applied.
    """
    n_spatial_dims = x.ndim - 1
    window_shape = _canon_int_sequence(window_shape, n_spatial_dims)
    activations = pool(
        x,
        0,
        lax.add,
        window_shape,
        strides,
        padding,
        input_dilation,
        window_dilation,
        data_format
    )
    return lax.div(
        activations,
        lax.convert_element_type(prod(window_shape), activations.dtype)
    )

def dot_product_attention_logits(
    query: Array, key: Array
) -> Array:
    """Compute scaled dot-product attention logits.
    
    :param query: Query array of shape
        ``(query_length, num_heads, query_key_depth)``.
    :param key: Key array of the same dtype as ``query`` and of shape
        ``(key_value_length, num_heads, query_key_depth)``.

    :returns: Attention logits of
        ``(num_heads, query_length, key_value_length)``.
    """
    logits = lax.dot_general(query, key, (((2,), (2,)), ((1,), (1,))))
    return lax.div(
        logits, lax.convert_element_type(sqrt(query.shape[2]), logits.dtype)
    )

def apply_attention_weights(
    value: Array, attention_weights: Array
) -> Array:
    """Apply attention weights to values.

    :param value: Value array of shape
        ``(key_value_length, num_heads, value_depth)``.
    :param attention_weights: Attention weights of the same dtype as ``value``
        and of shape ``(num_heads, query_length, key_value_length)``.

    :returns activations: ``value`` with ``attention_weights`` applied, of shape
        ``(query_length, num_heads, value_depth)``.
    """
    activations = lax.dot_general(
        value, attention_weights, (((0,), (2,)), ((1,), (0,)))
    )
    # activations: (num_heads, depth, value_length)
    return lax.transpose(activations, (2, 0, 1))

def z_norm(
    x: Array,
    axis: Union[str, int, Sequence[int]],
    batch_axis_name: Union[Hashable, Tuple[Hashable]]=(),
    epsilon: float=1e-05
):
    """Apply Z-score normalization.

    :param axis: "all", "channel_last", "channel_first", axis, or sequence of
        axes to normalize input features along. "all" indicates normalization
        along all axes (layer norm). "channel_last" and "channel_first" indicate
        normalization along all but the channel axis, assumed to be the last or
        first axis (instance norm).
    :param epsilon: Small number added to variance to avoid divisions by zero.
        Default: 1e-05.
    :param batch_axis_name: Hashable or tuple of hashable representing
        the batch axis name(s) to normalize along in addition to those in
        ``axis``. Default: (), no normlization along any batch axis.

    :returns: ``x`` with normalization applied.
    """
    if axis == "all":
        axis = list(range(x.ndim))
    elif axis == "channel_last":
        axis = list(range(x.ndim - 1))
    elif axis == "channel_first":
        axis = list(range(1, x.ndim))
    mean, variance = _compute_std_stats(x, axis, batch_axis_name)
    return _standadize(x, axis, mean, variance, epsilon)
