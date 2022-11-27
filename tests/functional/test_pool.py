from mlax.functional import avg_pool, max_pool, sum_pool
import jax.numpy as jnp
from jax import (
    lax
)

dtype = jnp.float16
inputs = jnp.ones((2, 4, 4), dtype)
window_shape = (1, 2, 2)

def test_max_pool():
    activations = max_pool(
        inputs,
        window_shape
    )
    assert lax.eq(
        activations,
        jnp.ones((2, 3, 3), dtype)
    ).all()

def test_sum_pool():
    activations = sum_pool(
        inputs,
        window_shape
    )
    assert lax.eq(
        activations,
        jnp.full((2, 3, 3), 4, dtype)
    ).all()

def test_avg_pool():
    activations = avg_pool(
        inputs,
        window_shape
    )
    assert lax.eq(
        activations,
        jnp.ones((2, 3, 3), dtype)
    ).all()

