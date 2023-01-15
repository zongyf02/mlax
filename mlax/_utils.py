from jax import lax
from math import prod
from inspect import signature

def _canon_precision(precision):
    return lax.Precision(precision)

def _canon_precision_pair(precision):
    if isinstance(precision, tuple):
        return _canon_precision(precision[0]), _canon_precision(precision[1])
    else:
        precision = _canon_precision(precision)
        return precision, precision

def _canon_dtype(dtype):
    return lax.dtype(dtype)

def _canon_opt_dtype(dtype):
    return None if dtype is None else _canon_dtype(dtype)

def _canon_int_sequence(int_or_seq, length):
    return (
        tuple([int_or_seq] * length) if isinstance(int_or_seq, int)
        else tuple(int_or_seq)
    )

def _canon_opt_int_sequence(opt_int_or_seq, length):
    return (
        None if opt_int_or_seq is None else
        _canon_int_sequence(opt_int_or_seq, length)
    )

def _canon_padding(padding, n_spatial_dims):
    if padding == "VALID":
        return tuple([(0, 0)] * n_spatial_dims)
    elif isinstance(padding, str):
        return padding
    elif isinstance(padding, int):
        return tuple([(padding, padding)] * n_spatial_dims)
    else:
        return tuple(
            (dim, dim) if isinstance(dim, int) else dim for dim in padding
        )

def _needs_rng(fwd):
    # Raises exception if ``fwd`` does not have the ``rng`` keyword param
    return signature(fwd).parameters["rng"].default is not None

def _n_elems(x, dims):
    return lax.convert_element_type(
        prod(d for i, d in enumerate(x.shape) if i in dims),
        x.dtype
    )

def _mean(x, dims, n_elems=None):
    if n_elems is None:
        n_elems = _n_elems(x, dims)
    return lax.div(
        lax.reduce(x, 0, lax.add, dims),
        n_elems
    )

def _variance(x, dims, mean=None, mean_of_squares=None, n_elems=None):
    if mean is None:
        if n_elems is None:
            n_elems = _n_elems(x, dims)
        mean = _mean(x, dims, n_elems)

    if mean_of_squares is None:
        if n_elems is None:
            n_elems = _n_elems(x, dims)
        mean_of_squares = _mean(lax.integer_pow(x, 2), dims, n_elems)

    return lax.sub(
        mean_of_squares,
        lax.integer_pow(mean, 2) # Square of means
    )

def _normalize(x, dims, eps=1e-5, mean=None, variance=None, n_elems=None):
    if mean is None:
        if n_elems is None:
            n_elems = _n_elems(x, dims)
        mean = _mean(x, dims, n_elems)

    if variance is None:
        variance = _variance(x, dims, mean=mean, n_elems=n_elems)

    broadcast_dims = [i for i in range(x.ndim) if i not in dims]

    return lax.mul(
        lax.sub(
            x,
            lax.broadcast_in_dim(mean, x.shape, broadcast_dims)
        ),
        lax.broadcast_in_dim(
            lax.rsqrt(
                lax.add(
                    variance,
                    lax.convert_element_type(
                        eps, x.dtype
                    )
                )
            ),
            x.shape, broadcast_dims 
        )
    )