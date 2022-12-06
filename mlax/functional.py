import jax
from jax import (
    random,
    lax
)
from mlax._utils import (
    _canon_int_sequence,
    _canon_opt_int_sequence,
    _canon_padding
)
from math import prod
from typing import Any, Sequence, Union, Callable, Optional

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
    window_shape: Sequence[int],
    strides: Union[int, Sequence[int]] = 1,
    padding="VALID",
    input_dilation: Optional[Union[int, Sequence[int]]] = None,
    window_dilation: Optional[Union[int, Sequence[int]]] = None
) -> jax.Array:
    """Apply an arbitrary reduce function over poolings windows of input
        features.

    :param x: Input features to the pooling transform.
    :param init_value: Initial value of the reduce function over each pooling
        window.
    :param reduce_fn: Reduce function.
    :param window_shape: Shape of the pooling window.
    :param strides: An integer or sequence integers of the same length as
        ``window_shape``, specifying the strides of the pooling window along the
        window shape. A single integer specifies the same value for all window
        dimensions. Default: 1.
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
    
    :returns y: x with pooling applied.
    
    .. _jax.lax.reduce_window:
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.reduce_window.html
    """
    window_ndims = len(window_shape)
    return lax.reduce_window(
        x,
        init_value,
        reduce_fn,
        window_shape,
        _canon_int_sequence(strides, window_ndims),
        _canon_padding(padding, window_ndims),
        _canon_opt_int_sequence(input_dilation, window_ndims),
        _canon_opt_int_sequence(window_dilation, window_ndims)
    )

def max_pool(
    x: jax.Array,
    window_shape: Sequence[int],
    strides: Union[int, Sequence[int]] = 1,
    padding="VALID",
    input_dilation=None,
    window_dilation=None
) -> jax.Array:
    """Apply max pooling over input features.

    :param x: Input features to the max pooling transform.
    :param window_shape: See the ``window_shape`` parameter of ``pooling``.
    :param strides: See the ``strides`` parameter of ``pooling``. Default: 1.
    :param padding: See the ``padding`` parameter of ``pooling``.
    :param input_dilation: See the ``input_dilation`` parameter of ``pooling``.
        Default: None, no input dilation.
    :param window_dilation: See the ``window_dilation`` parameter of
        ``pooling``. Default: None, no window dilation.
    
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
        window_dilation
    )

def sum_pool(
    x: jax.Array,
    window_shape: Sequence[int],
    strides: Union[int, Sequence[int]] = 1,
    padding="VALID",
    input_dilation=None,
    window_dilation=None
) -> jax.Array:
    """Apply sum pooling over input features.

    :param x: Input features to the sum pooling transform.
    :param window_shape: See the ``window_shape`` parameter of ``pooling``.
    :param strides: See the ``strides`` parameter of ``pooling``. Default: 1.
    :param padding: See the ``padding`` parameter of ``pooling``.
    :param input_dilation: See the ``input_dilation`` parameter of ``pooling``.
        Default: None, no input dilation.
    :param window_dilation: See the ``window_dilation`` parameter of
        ``pooling``. Default: None, no window dilation.
    
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
        window_dilation
    )

def avg_pool(
    x: jax.Array,
    window_shape: Sequence[int],
    strides: Union[int, Sequence[int]] = 1,
    padding="VALID",
    input_dilation=None,
    window_dilation=None
) -> jax.Array:
    """Apply average pooling over input features.

    :param x: Input features to the avg pooling transform.
    :param window_shape: See the ``window_shape`` parameter of ``pooling``.
    :param strides: See the ``strides`` parameter of ``pooling``. Default: 1.
    :param padding: See the ``padding`` parameter of ``pooling``.
    :param input_dilation: See the ``input_dilation`` parameter of ``pooling``.
        Default: None, no input dilation.
    :param window_dilation: See the ``window_dilation`` parameter of
        ``pooling``. Default: None, no window dilation.
 
    :returns y: x with average pooling applied.
    """
    activations = pool(
        x,
        0,
        lax.add,
        window_shape,
        strides,
        padding,
        input_dilation,
        window_dilation
    )
    return lax.div(
        activations,
        lax.convert_element_type(prod(window_shape), activations.dtype)
    )
