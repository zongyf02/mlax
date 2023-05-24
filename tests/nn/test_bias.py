from jax import (
    numpy as jnp,
    random,
    nn
)
import pytest
from mlax.nn import Bias
from mlax._test_utils import layer_test_results, assert_equal_array

@pytest.mark.parametrize(
    "config,x,expected_output,expected_bias_kernel",
    [
        (
            {
                "rng": random.PRNGKey(0),
                "in_features": (0, 1, -1),
                "bias_initializer": nn.initializers.constant(1, jnp.float16),
                "dtype": jnp.float32
            },
            random.normal(random.PRNGKey(7), (2, 4, 4, 3), jnp.bfloat16),
            random.normal(random.PRNGKey(7), (2, 4, 4, 3), jnp.bfloat16) + 1,
            jnp.ones((1, 3), jnp.float32)
        ),
        (
            {
                "rng": random.PRNGKey(1),
                "in_features": (),
                "bias_initializer": nn.initializers.constant(2, jnp.float16),
                "dtype": jnp.float32
            },
            random.normal(random.PRNGKey(7), (2, 4, 4, 3), jnp.bfloat16),
            random.normal(random.PRNGKey(7), (2, 4, 4, 3), jnp.bfloat16) + 2,
            jnp.full((), 2, jnp.float32)
        ),
        (
            {
                "rng": random.PRNGKey(2),
                "in_features": -1,
                "bias_initializer": nn.initializers.constant(3, jnp.float16),
                "dtype": jnp.bfloat16
            },
            random.normal(random.PRNGKey(7), (2, 8), jnp.float32),
            random.normal(random.PRNGKey(7), (2, 8), jnp.float32) + 3,
            jnp.full((8,), 3, jnp.bfloat16)
        ),
        (
            {
                "rng": random.PRNGKey(3),
                "in_features": [],
                "bias_initializer": nn.initializers.constant(1, jnp.float16),
                "dtype": jnp.bfloat16
            },
            random.normal(random.PRNGKey(7), (4, 4), jnp.float32),
            random.normal(random.PRNGKey(7), (4, 4), jnp.float32) + 1,
            jnp.ones((), dtype=jnp.bfloat16)
        )
    ]
)
def test_bias(config, x, expected_output, expected_bias_kernel):
    layer, (t_acts, new_t_layer), (i_acts, new_i_layer) = layer_test_results(
        Bias, config, x
    )
    assert_equal_array(layer.bias_kernel.data, expected_bias_kernel)

    assert_equal_array(t_acts, expected_output)
    assert_equal_array(new_t_layer.bias_kernel.data, expected_bias_kernel)

    assert_equal_array(i_acts, expected_output)
    assert_equal_array(new_i_layer.bias_kernel.data, expected_bias_kernel)
