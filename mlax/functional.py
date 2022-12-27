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

    :param x: Input features to the pooling transform.
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

    :param x: Input features to the max pooling transform.
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

    :param x: Input features to the sum pooling transform.
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

    :param x: Input features to the avg pooling transform.
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
