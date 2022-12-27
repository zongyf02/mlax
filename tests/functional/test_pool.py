from mlax.functional import avg_pool, max_pool, sum_pool, avg_pool
import jax.numpy as jnp
from jax import (
    lax
)

dtype = jnp.float16
inputs1 = jnp.ones((2, 1, 4, 4, 4), dtype)
inputs2 = jnp.ones((2, 4, 4, 4, 1), dtype)

def test_max_pool():
    activations = max_pool(
        inputs1,
        ndims=3,
        window_shape=2,
        padding=0
    )
    assert lax.eq(
        activations,
        jnp.ones((2, 1, 3, 3, 3), dtype)
    ).all()

    activations = max_pool(
        inputs2,
        ndims=3,
        window_shape=(2, 2, 2),
        strides=(1, 1, 1),
        channel_last=True
    )
    assert lax.eq(
        activations,
        jnp.ones((2, 3, 3, 3, 1), dtype)
    ).all()

def test_sum_pool():
    activations = sum_pool(
        inputs1,
        ndims=3,
        window_shape=2,
        padding=0,
        input_dilation=1
    )
    assert lax.eq(
        activations,
        jnp.full((2, 1, 3, 3, 3), 8, dtype)
    ).all()

    activations = sum_pool(
        inputs2,
        ndims=3,
        window_shape=(2, 2, 2),
        strides=(1, 1, 1),
        input_dilation=1,
        channel_last=True
    )
    assert lax.eq(
        activations,
        jnp.full((2, 3, 3, 3, 1), 8, dtype)
    ).all()

def test_avg_pool():
    activations = avg_pool(
        inputs1,
        ndims=3,
        window_shape=2,
        window_dilation=1,
        padding=0
    )
    assert lax.eq(
        activations,
        jnp.ones((2, 1, 3, 3, 3), dtype)
    ).all()

    activations = avg_pool(
        inputs2,
        ndims=3,
        window_shape=(2, 2, 2),
        strides=(1, 1, 1),
        window_dilation=1,
        channel_last=True
    )
    assert lax.eq(
        activations,
        jnp.ones((2, 3, 3, 3, 1), dtype)
    ).all()

