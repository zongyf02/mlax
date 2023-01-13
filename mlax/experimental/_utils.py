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
        [int_or_seq] * length if isinstance(int_or_seq, int) else
        list(int_or_seq)
    )

def _canon_opt_int_sequence(opt_int_or_seq, length):
    return (
        None if opt_int_or_seq is None else
        _canon_int_sequence(opt_int_or_seq, length)
    )

def _canon_padding(padding, n_spatial_dims):
    return (
        padding if isinstance(padding, str) else
        [(padding, padding)] * n_spatial_dims if isinstance(padding, int) else
        list(
            (dim, dim) if isinstance(dim, int) else dim for dim in padding
        )
    )

def _needs_rng(fwd):
    # Raises exception if ``fwd`` does not have the ``rng`` keyword param
    return signature(fwd).parameters["rng"].default is not None
