from jax import lax
import sys
from inspect import signature

def _canon_precision(precision):
    if precision is None or isinstance(precision, lax.Precision):
        return precision
    elif isinstance(precision, tuple):
        return tuple(_canon_precision(arg) for arg in precision)
    else:
        return lax.Precision(precision)

def _canon_opt_dtype(dtype):
    return None if dtype is None else lax.dtype(dtype)

def _canon_int_sequence(int_or_seq, length):
    return (
        (int_or_seq,) * length if isinstance(int_or_seq, int) else
        tuple(int_or_seq)
    )

def _canon_opt_int_sequence(opt_int_or_seq, length):
    return (
        None if opt_int_or_seq is None else
        (opt_int_or_seq,) * length if isinstance(opt_int_or_seq, int) else
        tuple(opt_int_or_seq)
    )

def _canon_padding(padding, ndims):
    return (
        padding if isinstance(padding, str) else
        ((padding, padding),) * ndims if isinstance(padding, int) else
        tuple(
            (dim, dim) if isinstance(dim, int) else dim for dim in padding
        )
    )

def _get_fwd(hyperparams):
    return sys.modules[hyperparams.__module__].fwd

def _needs_key(fwd):
    return signature(fwd).parameters.__contains__("key")
