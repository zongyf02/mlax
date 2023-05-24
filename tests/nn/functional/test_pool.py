import jax.numpy as jnp
from jax import lax
import pytest
from mlax.nn.functional import (
    avg_pool, max_pool, sum_pool, avg_pool
)
from mlax._test_utils import assert_equal_array

@pytest.mark.parametrize(
    "x,params,expected_output",
    [
        (
            jnp.ones((1, 4, 4, 4), jnp.float32),
            {
                "window_shape": 2,
                "padding": 0,
                "input_dilation": 1,
                "data_format": "channel_first"
            },
            jnp.ones((1, 3, 3, 3), jnp.float32)
        ),
        (
            jnp.ones((4, 4, 4, 1), jnp.float16),
            {
                "window_shape": (2, 2, 2),
                "strides": (1, 1, 1)
            },
            jnp.ones((3, 3, 3, 1), jnp.float16)
        ),
        (
            jnp.ones((4, 4, 1, 4), jnp.bfloat16),
            {
                "window_shape": (2, 2, 2),
                "window_dilation": 1,
                "data_format": "HWCD"
            },
            jnp.ones((3, 3, 3, 1), jnp.bfloat16)
        )
    ]
)
def test_max_pool(x, params, expected_output):
    activations = max_pool(x, **params)
    assert_equal_array(activations, expected_output)

@pytest.mark.parametrize(
    "x,params,expected_output",
    [
        (
            jnp.ones((1, 4, 4, 4), jnp.bfloat16),
            {
                "window_shape": 2,
                "padding": 0,
                "input_dilation": 1,
                "data_format": "channel_first"
            },
            jnp.full((1, 3, 3, 3), 8, jnp.bfloat16)
        ),
        (
            jnp.ones((4, 4, 4, 1), jnp.float32),
            {
                "window_shape": (2, 2, 2),
                "strides": (1, 1, 1),
                "window_dilation": 1
            },
            jnp.full((3, 3, 3, 1), 8, jnp.float32)
        ),
        (
            jnp.ones((4, 4, 1, 4), jnp.int8),
            {
                "window_shape": 2,
                "padding": (0, 0, 0),
                "input_dilation": 1,
                "window_dilation": 1,
                "data_format": "HWCD"
            },
            jnp.full((3, 3, 3, 1), 8, jnp.int8)
        )
    ]
)
def test_sum_pool(x, params, expected_output):
    activations = sum_pool(x, **params)
    assert_equal_array(activations, expected_output)

@pytest.mark.parametrize(
    "x,params,expected_output",
    [
        (
            jnp.ones((1, 4, 4, 4), jnp.float16),
            {
                "window_shape": 2,
                "padding": 0,
                "window_dilation": 1,
                "data_format": "channel_first"
            },
            jnp.ones((1, 3, 3, 3), jnp.float16)
        ),
        (
            jnp.ones((4, 4, 4, 1), jnp.bfloat16),
            {
                "window_shape": (2, 2, 2),
                "padding": (0, 0, 0),
                "strides": (1, 1, 1),
            },
            jnp.ones((3, 3, 3, 1), jnp.bfloat16)
        ),
        (
            jnp.ones((4, 4, 1, 4), jnp.float32),
            {
                "window_shape": 2,
                "input_dilation": 1,
                "window_dilation": 1,
                "data_format": "HWCD"
            },
            jnp.ones((3, 3, 3, 1), jnp.float32)
        )
    ]
)
def test_avg_pool(x, params, expected_output):
    activations = avg_pool(x, **params)
    assert_equal_array(activations, expected_output)
