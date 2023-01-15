import jax
from jax import (
    random,
    lax
)
from mlax._utils import (
    _canon_int_sequence,
    _canon_opt_int_sequence,
    _canon_padding,
    _normalize
)
from math import prod, sqrt
from typing import Any, Tuple, Sequence, Union, Callable, Optional

def identity(x: jax.Array) -> jax.Array:
    """Identity function.
    
    :param x: Input features.

    :returns y: ``x``.
    """
    return x

def dropout(
    x: jax.Array,
    rng: Any,
    prob: float
) -> jax.Array:
    """Apply random dropouts to input features.

    :param x: Input features to the dropout transform.
    :param rng: PRNG key for randomizing dropouts.
    :param prob: Probability at which each element is kept. Must be of a
        non-zero floating point type.

    :returns y: x with dropouts applied.
    """
    mask = random.bernoulli(rng, prob, x.shape)
    zeros = lax.full_like(x, 0)
    return lax.select(
        mask,
        lax.div(x, lax.convert_element_type(prob, x.dtype)),
        zeros
    )

def pool(
    x: jax.Array,
    init_value: Any,
    reduce_fn: Callable[[Any, Any], Any],
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]] = 1,
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "VALID",
    input_dilation: Optional[Union[int, Sequence[int]]] = None,
    window_dilation: Optional[Union[int, Sequence[int]]] = None,
    channel_last: bool=False
) -> jax.Array:
    """Apply an arbitrary reduce function over poolings windows of input
        features.

    :param x: Input features to the avg pooling transform that have
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
    :param channel_last: Whether features are channel-last or first. Default:
        False, channel-first.
    
    :returns y: x with pooling applied.
    
    .. _jax.lax.reduce_window:
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.reduce_window.html
    """
    n_spatial_dims = x.ndim - 1
    window_shape = _canon_int_sequence(window_shape, n_spatial_dims)
    strides = _canon_int_sequence(strides, n_spatial_dims)
    padding = _canon_padding(padding, n_spatial_dims)
    input_dilation = _canon_opt_int_sequence(input_dilation, n_spatial_dims)
    window_dilation = _canon_opt_int_sequence(window_dilation, n_spatial_dims)

    if channel_last:
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
    else:
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
    x: jax.Array,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]] = 1,
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "VALID",
    input_dilation: Optional[Union[int, Sequence[int]]] = None,
    window_dilation: Optional[Union[int, Sequence[int]]] = None,
    channel_last: bool=False
) -> jax.Array:
    """Apply max pooling over input features.

    :param x: Input features to the avg pooling transform. Must be compatible
        with ``channel_last``.
    :param window_shape: See the ``window_shape`` parameter of ``pooling``.
    :param strides: See the ``strides`` parameter of ``pooling``. Default: 1.
    :param padding: See the ``padding`` parameter of ``pooling``.
    :param input_dilation: See the ``input_dilation`` parameter of ``pooling``.
        Default: None, no input dilation.
    :param window_dilation: See the ``window_dilation`` parameter of
        ``pooling``. Default: None, no window dilation.
    :param channel_last: Whether features are channel-last or first. Default:
        False, channel-first.
    
    :returns y: x with max pooling applied.
    """
    return pool(
        x,
        -jax.numpy.inf,
        lax.max,
        window_shape,
        strides,
        padding,
        input_dilation,
        window_dilation,
        channel_last
    )

def sum_pool(
    x: jax.Array,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]] = 1,
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "VALID",
    input_dilation: Optional[Union[int, Sequence[int]]] = None,
    window_dilation: Optional[Union[int, Sequence[int]]] = None,
    channel_last: bool=False
) -> jax.Array:
    """Apply sum pooling over input features.

    :param x: Input features to the avg pooling transform. Must be compatible
        with ``channel_last``.
    :param window_shape: See the ``window_shape`` parameter of ``pooling``.
    :param strides: See the ``strides`` parameter of ``pooling``. Default: 1.
    :param padding: See the ``padding`` parameter of ``pooling``.
    :param input_dilation: See the ``input_dilation`` parameter of ``pooling``.
        Default: None, no input dilation.
    :param window_dilation: See the ``window_dilation`` parameter of
        ``pooling``. Default: None, no window dilation.
    :param channel_last: Whether features are channel-last or first. Default:
        False, channel-first.
    
    :returns y: x with sum pooling applied.
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
        channel_last
    )

def avg_pool(
    x: jax.Array,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]] = 1,
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "VALID",
    input_dilation: Optional[Union[int, Sequence[int]]] = None,
    window_dilation: Optional[Union[int, Sequence[int]]] = None,
    channel_last: bool=False
) -> jax.Array:
    """Apply average pooling over input features.

    :param x: Input features to the avg pooling transform. Must be compatible
        with ``channel_last``.
    :param window_shape: See the ``window_shape`` parameter of ``pooling``.
    :param strides: See the ``strides`` parameter of ``pooling``. Default: 1.
    :param padding: See the ``padding`` parameter of ``pooling``.
    :param input_dilation: See the ``input_dilation`` parameter of ``pooling``.
        Default: None, no input dilation.
    :param window_dilation: See the ``window_dilation`` parameter of
        ``pooling``. Default: None, no window dilation.
    :param channel_last: Whether features are channel-last or first. Default:
        False, channel-first.
 
    :returns y: x with average pooling applied.
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
        channel_last
    )

    return lax.div(
        activations,
        lax.convert_element_type(prod(window_shape), activations.dtype)
    )

def dot_product_attention_logits(
    query: jax.Array,
    key: jax.Array
):
    """Compute dot-product attention logits.
    
    :param query: Query array of shape ``(query_length, num_heads, depth)``.
    :param key: Key array of the same dtype as ``query`` and of shape
        ``(key_length, num_heads, depth)``.

    :returns: Attention logits of ``(num_heads, query_length, key_length)``.
    """
    logits = lax.dot_general(
        query, key,
        (((2,), (2,)), ((1,), (1,)))
    )
    return lax.div(
        logits,
        lax.convert_element_type(sqrt(query.shape[-1]), logits.dtype)
    )

def apply_attention_mask(
    logits: jax.Array,
    mask: jax.Array,
    masked_value=-jax.numpy.inf
):
    """Apply attention mask to logits.

    :param logits: Attention logits of shape
        ``(num_heads, query_length, key_length)``.
    :param mask: Mask array of same shape as ``logits``. Must be boolean or
        integer type.
    :param masked_value: Value that will be taken by the masked logits. Default:
        -inf, minimum value allowed by ``logits``' type.
    
    :returns logits: ``logits`` with ``mask`` applied.
    """
    masked_value = lax.full_like(
        logits,
        lax.convert_element_type(masked_value, logits.dtype),
    )
    return lax.select(mask, logits, masked_value)

def apply_attention_weights(
    value: jax.Array,
    attention_weights: jax.Array
):
    """Apply attention weights to values.

    :param value: Value array of shape ``(value_length, num_heads, depth)``.
    :param attention_weights: Attention weights of the same dtype as ``value``
        and of shape ``(num_heads, query_length, value_length)``.

    :returns activations: ``value`` with ``attention_weights`` applied, of shape
        ``(value_length, num_heads, depth)``.
    """
    activations = lax.dot_general(
        value, attention_weights,
        (((0,), (2,)), ((1,), (0,)))
    )
    # activations: (num_heads, depth, value_length)
    return lax.transpose(activations, (2, 0, 1))

def layer_norm(x, epsilon=1e-05):
    """Apply layer normalization.

    :param x: Input features to layer norm, either in channel-first or
        channel-last format.
    :param epsilon: Small number added to variance to avoid divisions by zero.
        Default: 1e-05.
    
    :returns: ``x`` with layer normalization applied.
    """
    return _normalize(x, range(x.ndim), epsilon)

def instance_norm(x, channel_last=False, epsilon=1e-05):
    """Apply instance normalization.

    :param x: Input features to layer norm. Must be compatible with
        ``channel_last``.
    :param channel_last: Whether features are channel-last or first. Default:
        False, channel-first.
    :param epsilon: Small number added to variance to avoid divisions by zero.
        Default: 1e-05.
    
    :returns: ``x`` with instance normalization applied.
    """
    dims = range(x.ndim - 1) if channel_last else range(1, x.ndim)
    return _normalize(
        x,
        dims,
        epsilon
    )

def group_norm(x, num_groups, channel_last=False, epsilon=1e-05):
    """Apply group normalization.

    :param x: Input features to layer norm. Must be compatible with
        ``channel_last``.
    :param num_groups: Number of groups to split the channels into. Must divide
        the number of channels in ``x``.
    :param channel_last: Whether features are channel-last or first. Default:
        False, channel-first.
    :param epsilon: Small number added to variance to avoid divisions by zero.
        Default: 1e-05.
    
    :returns: ``x`` with group normalization applied.
    """
    x_shape = x.shape
    if channel_last:
        num_channels = x_shape[-1]
        x = lax.reshape(
            x, (*x_shape[:-1], num_channels//num_groups, num_groups)
        )
        dims = range(x.ndim - 1)
    else:
        num_channels = x_shape[0]
        x = lax.reshape(
            x, (num_groups, num_channels//num_groups, *x_shape[1:])
        )
        dims = range(1, x.ndim)
    
    return lax.reshape(
        _normalize(
            x,
            dims,
            epsilon
        ),
        x_shape
    )