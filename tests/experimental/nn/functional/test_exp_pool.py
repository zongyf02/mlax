from mlax.experimental.nn.functional import (
    avg_pool, max_pool, sum_pool, avg_pool
)
import jax.numpy as jnp
from jax import (
    lax
)
import pytest

@pytest.mark.parametrize(
    "input,params,expected_output",
    [
        (
            jnp.ones((1, 4, 4, 4), jnp.float16),
            {
                "window_shape": 2,
                "padding": 0
            },
            jnp.ones((1, 3, 3, 3), jnp.float16)
        ),
        (
            jnp.ones((4, 4, 4, 1), jnp.float16),
            {
                "window_shape": (2, 2, 2),
                "strides": (1, 1, 1),
                "channel_last": True
            },
            jnp.ones((3, 3, 3, 1), jnp.float16)
        )
    ]
)
def test_max_pool(input, params, expected_output):
    activations = max_pool(
        input,
        **params
    )
    assert lax.eq(
        activations,
        expected_output
    ).all()

@pytest.mark.parametrize(
    "input,params,expected_output",
    [
        (
            jnp.ones((1, 4, 4, 4), jnp.float16),
            {
                "window_shape": 2,
                "padding": 0,
                "input_dilation": 1
            },
            jnp.full((1, 3, 3, 3), 8, jnp.float16)
        ),
        (
            jnp.ones((4, 4, 4, 1), jnp.float16),
            {
                "window_shape": (2, 2, 2),
                "strides": (1, 1, 1),
                "input_dilation": 1,
                "channel_last": True
            },
            jnp.full((3, 3, 3, 1), 8, jnp.float16)
        )
    ]
)
def test_sum_pool(input, params, expected_output):
    activations = sum_pool(
        input,
        **params
    )
    assert lax.eq(
        activations,
        expected_output
    ).all()

@pytest.mark.parametrize(
    "input,params,expected_output",
    [
        (
            jnp.ones((1, 4, 4, 4), jnp.float16),
            {
                "window_shape": 2,
                "padding": 0,
                "window_dilation": 1
            },
            jnp.ones((1, 3, 3, 3), jnp.float16)
        ),
        (
            jnp.ones((4, 4, 4, 1), jnp.float16),
            {
                "window_shape": (2, 2, 2),
                "strides": (1, 1, 1),
                "window_dilation": 1,
                "channel_last": True
            },
            jnp.ones((3, 3, 3, 1), jnp.float16)
        )
    ]
)
def test_avg_pool(input, params, expected_output):
    activations = avg_pool(
        input,
        **params
    )
    assert lax.eq(
        activations,
        expected_output
    ).all()
