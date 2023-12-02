import math
from typing import Optional
from functools import partial
import jax
from jax import (
    Array,
    lax,
    nn,
    numpy as jnp
)
from mlax.nn.functional import (
    dot_product_attention_logits,
    apply_attention_weights
)

@partial(jax.vmap, in_axes=(1, 1, 1, None), out_axes=1)
def attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Optional[Array]
):
    # logits: (q_length, kv_length)
    logits = dot_product_attention_logits(query, key)
    if mask is None:
        logits = jnp.where(mask, logits, -jnp.inf)

    # logits, weights: (q_length, kv_length)
    weights = nn.softmax(logits)

    # activations: (q_length, v_head_depth)
    activations = apply_attention_weights(weights, value)
    return activations

def _linear_attention_chunk(
    query_chunk: Array,
    key: Array,
    value: Array,
    mask_slice: Optional[Array],
    key_value_chunk_size: int,
    precision,
    dtype
) -> Array:
    kv_len, n_heads, d_head = key.shape
    key_value_chunk_size = min(key_value_chunk_size, kv_len)

    query_chunk = query_chunk / jnp.sqrt(d_head).astype(dtype)

    @partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(query_chunk, key_chunk, value_chunk,
                        mask_chunk: Optional[Array]):
        attn_weights = jnp.einsum("...qhd,...khd->...hqk",
                                  query_chunk,
                                  key_chunk,
                                  precision=precision).astype(dtype)

        if mask_chunk is not None:
            big_neg = jnp.finfo(dtype).min
            attn_weights = jnp.where(mask_chunk, attn_weights, big_neg)

        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = lax.stop_gradient(max_score)

        exp_weights = jnp.exp(attn_weights - max_score)

        exp_values = jnp.einsum("...hqk,...khd->...qhd",
                                exp_weights,
                                value_chunk,
                                precision=precision).astype(dtype)

        return (exp_values, exp_weights.sum(axis=-1), max_score.squeeze(-1))

    def chunk_scanner(chunk_idx):
        key_chunk = lax.dynamic_slice(key, (chunk_idx, 0, 0),
                                      slice_sizes=(key_value_chunk_size,
                                                   n_heads, d_head))

        value_chunk = lax.dynamic_slice(value, (chunk_idx, 0, 0),
                                        slice_sizes=(key_value_chunk_size,
                                                     n_heads, d_head))

        if mask_slice is not None:
            mask_chunk = lax.dynamic_slice(mask_slice, (0, 0, chunk_idx),
                                           slice_sizes=(1, mask_slice.shape[1],
                                                        key_value_chunk_size))
        else:
            mask_chunk = None

        return summarize_chunk(query_chunk, key_chunk, value_chunk, mask_chunk)

    chunk_exp_values, chunk_exp_attn_weights, chunk_max_scores = lax.map(
        chunk_scanner, xs=jnp.arange(0, kv_len, key_value_chunk_size))

    chunk_exp_attn_weights = jnp.transpose(chunk_exp_attn_weights, (0, 2, 1))
    chunk_max_scores = jnp.transpose(chunk_max_scores, (0, 2, 1))

    global_max = jnp.max(chunk_max_scores, axis=0, keepdims=True)
    max_score_diffs = jnp.exp(chunk_max_scores - global_max)

    chunk_exp_values *= jnp.expand_dims(max_score_diffs, axis=-1)
    chunk_exp_attn_weights *= max_score_diffs

    exp_values = chunk_exp_values.sum(axis=0)
    exp_attn_weights = jnp.expand_dims(chunk_exp_attn_weights,
                                       axis=-1).sum(axis=0)

    return exp_values / exp_attn_weights

def _linear_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Optional[Array],
    query_chunk_size: int,
    key_value_chunk_size: int,
    precision=None,
    dtype=None
) -> Array:
    query_len, n_heads, d_head = query.shape

    def chunk_scanner(chunk_idx, _):
        query_chunk = lax.dynamic_slice(query, (chunk_idx, 0, 0),
                                        slice_sizes=(min(
                                            query_chunk_size,
                                            query_len), n_heads, d_head))

        if mask is not None:
            mask_slice = lax.dynamic_slice(
                mask, (0, chunk_idx, 0),
                slice_sizes=(1, min(query_chunk_size, query_len), query_len))
        else:
            mask_slice = None

        return (chunk_idx + query_chunk_size,
                _linear_attention_chunk(query_chunk, key, value, mask_slice,
                                        key_value_chunk_size, precision, dtype))

    _, out = lax.scan(chunk_scanner,
                      init=0,
                      xs=None,
                      length=math.ceil(query_len / query_chunk_size))

    return out.reshape((query_len, n_heads, d_head))

# Code from https://github.com/google-research/google-research/tree/master/memory_efficient_attention
# Linear-memory self-attention (actually sqrt(n)) using the online softmax normalization trick
def linear_attention(query: Array,
                     key: Array,
                     value: Array,
                     mask: Optional[Array]=None,
                     query_chunk_size: int=1024,
                     key_value_chunk_size:int=1024,
                     dtype= None,
                     precision = None
) -> Array:
    dtype = query.dtype

    return _linear_attention(query, key, value, mask, query_chunk_size,
                             key_value_chunk_size, precision, dtype)
