import pytest
from jax import (
    numpy as jnp,
    random,
    nn
)
from mlax.nn import Scaler
from mlax._test_utils import layer_test_results, assert_equal_array

@pytest.mark.parametrize(
    "config,x,expected_output,expected_scaler_kernel",
    [
        (
            {
                "rng": random.PRNGKey(0),
                "in_features": (0, 1, -1),
                "scaler_initializer": nn.initializers.constant(1, jnp.float16),
                "dtype": jnp.float32
            },
            random.normal(random.PRNGKey(7), (2, 4, 4, 3), jnp.bfloat16),
            random.normal(random.PRNGKey(7), (2, 4, 4, 3), jnp.bfloat16),
            jnp.ones((1, 3), jnp.float32)
        ),
        (
            {
                "rng": random.PRNGKey(1),
                "in_features": (),
                "scaler_initializer": nn.initializers.constant(2, jnp.float16),
                "dtype": jnp.float32
            },
            random.normal(random.PRNGKey(7), (2, 4, 4, 3), jnp.bfloat16),
            random.normal(random.PRNGKey(7), (2, 4, 4, 3), jnp.bfloat16) * 2,
            jnp.full((), 2, jnp.float32)
        ),
        (
            {
                "rng": random.PRNGKey(2),
                "in_features": -1,
                "scaler_initializer": nn.initializers.constant(3, jnp.float16),
                "dtype": jnp.bfloat16
            },
            random.normal(random.PRNGKey(7), (2, 8), jnp.float32),
            random.normal(random.PRNGKey(7), (2, 8), jnp.float32) * 3,
            jnp.full((8,), 3, jnp.bfloat16)
        ),
    ]
)
def test_scaling(config, x, expected_output, expected_scaler_kernel):
    layer, (t_acts, new_t_layer), (i_acts, new_i_layer) = layer_test_results(
        Scaler, config, x
    )
    assert_equal_array(layer.scaler_kernel.data, expected_scaler_kernel)

    assert_equal_array(t_acts, expected_output)
    assert_equal_array(new_t_layer.scaler_kernel.data, expected_scaler_kernel)

    assert_equal_array(i_acts, expected_output)
    assert_equal_array(new_i_layer.scaler_kernel.data, expected_scaler_kernel)
