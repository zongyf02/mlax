from jax import lax
from math import prod
import sys
from inspect import signature
from dataclasses import dataclass

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
        _canon_int_sequence(opt_int_or_seq, length)
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

_nn_hyperparams = dataclass(frozen=True, slots=True)

def _n_elems(x, reduce_dims):
    return lax.convert_element_type(
        prod(d for i, d in enumerate(x.shape) if i in reduce_dims),
        x.dtype
    )

def _mean(x, reduce_dims, n_elems):
    return lax.div(
        lax.reduce(x, 0, lax.add, reduce_dims),
        n_elems
    )

def _variance(x, reduce_dims, n_elems, mean):
    return lax.sub(
        lax.div(
            lax.reduce(
                lax.integer_pow(x, 2), # integer_pow not in lax docs
                0, lax.add, reduce_dims
            ),
            n_elems
        ),
        lax.integer_pow(mean, 2)
    )

def _normalize(x, broadcast_dims, eps, mean, variance):
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
