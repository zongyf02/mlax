from jax import (
    lax,
    dtypes
)
from math import prod
from inspect import signature

def _canon_precision_pair(precision):
    if isinstance(precision, tuple):
        return tuple(lax.Precision(p) for p in precision)
    else:
        precision = lax.Precision(precision)
        return precision, precision

def _canon_opt_dtype(dtype):
    return None if dtype is None else dtypes.canonicalize_dtype(dtype)

def _canon_int_sequence(int_or_seq, length=None):
    if isinstance(int_or_seq, int):
        if length is None:
            return int(int_or_seq)
        else:
            return tuple([int(int_or_seq)] * length)
    else:
        return tuple(int(i) for i in int_or_seq)

def _canon_opt_int_sequence(opt_int_or_seq, length=None):
    return (
        None if opt_int_or_seq is None else
        _canon_int_sequence(opt_int_or_seq, length)
    )

def _canon_padding(padding, n_spatial_dims=None):
    if isinstance(padding, str):
        return str(padding)
    elif isinstance(padding, int):
        if n_spatial_dims is None:
            return int(padding)
        else:
            return tuple([(int(padding), int(padding))] * n_spatial_dims)
    else:
        return tuple(
            (int(dim), int(dim)) if isinstance(dim, int)
            else tuple(int(i) for i in dim)
            for dim in padding
        )

def _canon_norm_axis(axis):
    if isinstance(axis, str):
        return str(axis)
    else:
        return _canon_int_sequence(axis, 1)

def _needs_rng(module):
    return signature(module.apply).parameters["rng"].default is not None

def _needs_axis_name(fn):
    return "axis_name" in signature(fn).parameters.keys()

def _compute_std_stats(x, axis, norm_axis_name=()):
    n_elems = lax.convert_element_type(lax.mul(
        prod(d for i, d in enumerate(x.shape) if i in axis),
        lax.psum(1, norm_axis_name)
    ), x.dtype)
    mean = lax.div(
        lax.psum(lax.reduce(x, 0, lax.add, axis), norm_axis_name), n_elems
    )
    mean_of_squares = lax.div(
        lax.psum(
            lax.reduce(lax.integer_pow(x, 2), 0, lax.add, axis), norm_axis_name
        ),
        n_elems
    )
    variance = lax.max(
        lax.convert_element_type(0, x.dtype),
        lax.sub(mean_of_squares, lax.integer_pow(mean, 2))
    )
    return mean, variance

def _standadize(x, axis, mean, variance, epsilon=1e-05):
    broadcast_dims = [i for i in range(x.ndim) if i not in axis]
    return lax.mul(
        lax.sub(x, lax.broadcast_in_dim(mean, x.shape, broadcast_dims)),
        lax.broadcast_in_dim(
            lax.rsqrt(
                lax.add(variance, lax.convert_element_type(epsilon, x.dtype))
            ),
            x.shape, broadcast_dims 
        )
    )
