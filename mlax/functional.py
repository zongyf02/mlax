import jax
from jax import (
    random,
    lax
)
from mlax._utils import (
    _canon_int_sequence,
    _canon_opt_int_sequence,
    _canon_padding,
    _n_elems,
    _mean,
    _variance,
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
    key: Any,
    prob: float,
    inference_mode: bool=False
) -> jax.Array:
    """Apply random dropouts to input features.

    :param x: Input features to the dropout transform.
    :param key: PRNG key for randomizing dropouts.
    :param prob: Probability at which each element is dropped. Must be of a
        non-zero floating point type.
    :param inference_mode: Whether in inference or training mode. When True,
        dropouts are not applied. Default: False.

    :returns y: x with dropouts applied.
    """
    if inference_mode:
        return x
    else:
        mask = random.bernoulli(key, prob, x.shape)
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
    ndims: int,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]] = 1,
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "VALID",
    input_dilation: Optional[Union[int, Sequence[int]]] = None,
    window_dilation: Optional[Union[int, Sequence[int]]] = None,
    channel_last: bool=False
) -> jax.Array:
    """Apply an arbitrary reduce function over poolings windows of input
        features.

    :param x: Batched input features to the avg pooling transform. Must be
        compatible with ``channel_last``.
    :param init_value: Initial value of the reduce function over each pooling
        window.
    :param reduce_fn: Reduce function.
    :param ndims: Number of input spatial dimensions.
    :param window_shape: An integer or a sequence of ``ndims`` integers,
        specifying the shape of the pooling window used on input features. A
        single integer specifies the same value for all spatial dimensions.
    :param strides: An integer or a sequence of ``ndims`` integers, specifying
        the strides of the window shape along the spatial dimensions. A single
        integer specifies the same value for all spatial dimensions. Default: 1.
    :param padding: String, integer, or a sequence of `ndims` integers or
        integer tuple pairs that give the padding to apply before and after
        each spatial dimension. If integer, the same padding is applied before
        and after all spatial dimensions. If a sequence of integers, then the
        same padding is applied before and after each spatial dimension.
        See the ``padding`` parameter of `jax.lax.reduce_window`_, which is used
        internally.
    :param input_dilation: None, an integer, or a sequence of ``ndims``
        integers, specifying the input dilation rate in each spatial dimension.
        See the ``base_dilation`` parameter of `jax.lax.reduce_window`_.
        Default: None, no input dilation.
    :param window_dilation: None, an integer, or a sequence of ``ndims``
        integers, specifying the window dilation rate in each spatial dimension.
        See the ``window_dilation`` parameter of `jax.lax.reduce_window`_.
        Default: None, no window dilation.
    :param channel_last: Whether features are channel-last or first. Default:
        False, channel-first.
    
    :returns y: x with pooling applied.
    
    .. _jax.lax.reduce_window:
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.reduce_window.html
    """
    window_shape = _canon_int_sequence(window_shape, ndims)
    strides = _canon_int_sequence(strides, ndims)
    padding = _canon_padding(padding, ndims)
    input_dilation = _canon_opt_int_sequence(input_dilation, ndims)
    window_dilation = _canon_opt_int_sequence(window_dilation, ndims)

    if channel_last:
        window_shape = (1, *window_shape, 1)
        strides = (1, *strides, 1)
        padding = (
            ((0, 0), *padding, (0, 0)) if isinstance(padding, tuple) else
            padding
        )
        input_dilation = (
            (1, *input_dilation, 1) if isinstance(input_dilation, tuple) else
            input_dilation
        )
        window_dilation = (
            (1, *window_dilation, 1) if isinstance(window_dilation, tuple) else
            window_dilation
        )
    else:
        window_shape = (1, 1, *window_shape)
        strides = (1, 1, *strides)
        padding = (
            ((0, 0), (0, 0), *padding) if isinstance(padding, tuple) else
            padding
        )
        input_dilation = (
            (1, 1, *input_dilation) if isinstance(input_dilation, tuple) else
            input_dilation
        )
        window_dilation = (
            (1, 1, *window_dilation) if isinstance(window_dilation, tuple) else
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
    ndims: int,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]] = 1,
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "VALID",
    input_dilation: Optional[Union[int, Sequence[int]]] = None,
    window_dilation: Optional[Union[int, Sequence[int]]] = None,
    channel_last: bool=False
) -> jax.Array:
    """Apply max pooling over input features.

    :param x: Batched input features to the avg pooling transform. Must be
        compatible with ``channel_last``.
    :param ndims: Number of input spatial dimensions.
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
        ndims,
        window_shape,
        strides,
        padding,
        input_dilation,
        window_dilation,
        channel_last
    )

def sum_pool(
    x: jax.Array,
    ndims: int,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]] = 1,
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "VALID",
    input_dilation: Optional[Union[int, Sequence[int]]] = None,
    window_dilation: Optional[Union[int, Sequence[int]]] = None,
    channel_last: bool=False
) -> jax.Array:
    """Apply sum pooling over input features.

    :param x: Batched input features to the avg pooling transform. Must be
        compatible with ``channel_last``.
    :param ndims: Number of input spatial dimensions.
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
        ndims,
        window_shape,
        strides,
        padding,
        input_dilation,
        window_dilation,
        channel_last
    )

def avg_pool(
    x: jax.Array,
    ndims: int,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]] = 1,
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "VALID",
    input_dilation: Optional[Union[int, Sequence[int]]] = None,
    window_dilation: Optional[Union[int, Sequence[int]]] = None,
    channel_last: bool=False
) -> jax.Array:
    """Apply average pooling over input features.

    :param x: Batched input features to the avg pooling transform. Must be
        compatible with ``channel_last``.
    :param ndims: Number of input spatial dimensions.
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
    activations = pool(
        x,
        0,
        lax.add,
        ndims,
        window_shape,
        strides,
        padding,
        input_dilation,
        window_dilation,
        channel_last
    )
    n_window_elems = (
        prod(window_shape) if isinstance(window_shape, tuple) else
        window_shape**ndims
    )
    return lax.div(
        activations,
        lax.convert_element_type(n_window_elems, activations.dtype)
    )

def layer_norm(x, epsilon=1e-05):
    """Apply layer normalization.

    :param x: Batched input features to layer norm, either in channel-first or
        channel-last format.
    :param epsilon: Small number added to variance to avoid divisions by zero.
        Default: 1e-05.
    
    :returns: ``x`` with layer normalization applied.
    """
    reduce_dims = range(1, len(x.shape))
    n_elems = _n_elems(x, reduce_dims)
    mean = _mean(x, reduce_dims, n_elems)
    variance = _variance(x, reduce_dims, n_elems, mean)
    return _normalize(x, (0,), epsilon, mean, variance)

def instance_norm(x, channel_last=False, epsilon=1e-05):
    """Apply instance normalization.

    :param x: Batched input features to layer norm. Must be compatible with
        ``channel_last``.
    :param channel_last: Whether features are channel-last or first. Default:
        False, channel-first.
    :param epsilon: Small number added to variance to avoid divisions by zero.
        Default: 1e-05.
    
    :returns: ``x`` with instance normalization applied.
    """
    in_ndims = len(x.shape)
    if channel_last:
        reduce_dims = range(1, in_ndims - 1)
        broadcast_dims = (0, in_ndims - 1)
    else:
        reduce_dims = range(2, in_ndims)
        broadcast_dims = (0, 1)

    n_elems = _n_elems(x, reduce_dims)
    mean = _mean(x, reduce_dims, n_elems)
    variance = _variance(x, reduce_dims, n_elems, mean)
    return _normalize(
        x,
        broadcast_dims,
        epsilon,
        mean,
        variance
    )

def group_norm(x, num_groups, channel_last=False, epsilon=1e-05):
    """Apply group normalization.

    :param x: Batched input features to layer norm. Must be compatible with
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
    in_ndims = len(x_shape)
    if channel_last:
        num_channels = x_shape[-1]
        x = lax.reshape(
            x, (*x_shape[:-1], num_groups, num_channels//num_groups)
        )
        reduce_dims = (*range(1, in_ndims - 1), in_ndims)
        broadcast_dims = (0, in_ndims - 1)
    else:
        num_channels = x_shape[1]
        x = lax.reshape(
            x, (x_shape[0], num_groups, num_channels//num_groups, *x_shape[2:])
        )
        reduce_dims = range(2, in_ndims + 1)
        broadcast_dims = (0, 1)
    
    n_elems = _n_elems(x, reduce_dims)
    mean = _mean(x, reduce_dims, n_elems)
    variance = _variance(x, reduce_dims, n_elems, mean)
    return lax.reshape(
        _normalize(
            x,
            broadcast_dims,
            epsilon,
            mean,
            variance
        ),
        x_shape
    )

def dot_product_attention_logits(
    query: jax.Array,
    key: jax.Array,
    scaled=True
):
    """Compute dot-product attention logits.
    
    :param query: Query array of shape
        ``(batch, query_length, num_heads, depth)``.
    :param key: Key array of the same dtype as ``query`` and of shape
        ``(batch, key_length, num_heads, depth)``.
    :param scaled: Whether to apply the scaling factor of ``1/sqrt(depth)``.
        Default: True.

    :returns: Attention logits of
        ``(batch, num_heads, query_length, key_length)``.
    """
    logits = lax.dot_general(
        query, key,
        (((3,), (3,)), ((0, 2), (0, 2)))
    )
    if scaled:
        scaler = lax.convert_element_type(sqrt(query.shape[-1]), logits.dtype)
        logits = lax.div(
            logits,
            scaler
        )
    return logits

def apply_attention_mask(logits, mask, masked_value=-jax.numpy.inf):
    """Apply attention mask to logits.

    :param logits: Attention logits of shape
        ``(batch, num_heads, query_length, key_length)``.
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

def apply_attention_weights(value, attention_weights):
    """Apply attention weights to values.

    :param value: Value array of shape
        ``(batch, value_length, num_heads, depth)``.
    :param attention_weights: Attention weights of the same dtype as ``value``
        and of shape ``(batch, num_heads, query_length, value_length)``.

    :returns activations: ``value`` with ``attention_weights`` applied, of shape
        ``(batch, value_length, num_heads, depth)``.
    """
    activations = lax.dot_general(
        value, attention_weights,
        (((1,), (3,)), ((0, 2), (0, 1)))
    )
    return lax.transpose(activations, (0, 3, 1, 2))
