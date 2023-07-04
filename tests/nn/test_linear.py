from jax import (
    numpy as jnp,
    random,
    nn,
    lax
)
import pytest
from mlax.nn import Linear
from mlax._test_utils import layer_test_results, assert_equal_array

@pytest.mark.parametrize(
    "config,x,expected_output,expected_linear_kernel",
    [
        (
            {
                "rng": random.PRNGKey(0),
                "out_features": 3,
                "precision": ("float32", "float32"),
                "transposed_kernel": False,
                "kernel_initializer": nn.initializers.constant(1, jnp.float16),
                "accum_dtype": jnp.float32,
                "dtype": jnp.float32
            },
            jnp.ones((2, 4), jnp.bfloat16),
            jnp.full((2, 3), 4, jnp.float32),
            jnp.ones((4, 3), jnp.float32)
        ),
        (
            {
                "rng": random.PRNGKey(1),
                "out_features": 3,
                "precision": lax.Precision.HIGHEST,
                "transposed_kernel": False,
                "kernel_initializer": nn.initializers.constant(1, jnp.float16),
                "accum_dtype": jnp.float32,
                "dtype": jnp.float32
            },
            jnp.ones((4, 5), jnp.bfloat16),
            jnp.full((4, 3), 5, jnp.float32),
            jnp.ones((5, 3), jnp.float32)
        ),
        (
            {
                "rng": random.PRNGKey(2),
                "out_features": 4,
                "precision": "float32",
                "transposed_kernel": True,
                "kernel_initializer": nn.initializers.constant(2, jnp.float16),
                "accum_dtype": jnp.float32,
                "dtype": jnp.bfloat16
            },
            jnp.ones((3, 4), jnp.bfloat16),
            jnp.full((3, 4), 8, jnp.float32),
            jnp.full((4, 4), 2, jnp.bfloat16)
        ),
    ]
)
def test_linear(config, x, expected_output, expected_linear_kernel):
    layer, (t_acts, new_t_layer), (i_acts, new_i_layer) = layer_test_results(
        Linear, config, x
    )
    assert_equal_array(layer.linear_kernel.data, expected_linear_kernel)

    assert_equal_array(t_acts, expected_output)
    assert_equal_array(new_t_layer.linear_kernel.data, expected_linear_kernel)

    assert_equal_array(i_acts, expected_output)
    assert_equal_array(new_i_layer.linear_kernel.data, expected_linear_kernel)
