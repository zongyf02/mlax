from jax import lax

def _canon_precision(precision):
    if precision is None or isinstance(precision, lax.Precision):
        return precision
    elif isinstance(precision, tuple):
        return tuple(_canon_precision(arg) for arg in precision)
    else:
        return lax.Precision(precision)

def _canon_dtype(dtype, param_dtype):
    return lax.dtype(param_dtype) if dtype is None else lax.dtype(dtype)

def _canon_accum_dtype(accum_dtype):
    return None if accum_dtype is None else lax.dtype(accum_dtype)

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
