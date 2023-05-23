from jax import (
    numpy as jnp,
    random,
    nn
)
import pytest
from mlax.nn import Conv
from mlax._test_utils import layer_test_results, assert_equal_array

@pytest.mark.parametrize(
    "config,x,expected_output,expected_conv_kernel",
    [
        (
            {
                "rng": random.PRNGKey(0),
                "out_channels": 16,
                "filter_shape": (5, 5),
                "strides": 1,
                "padding": 0,
                "input_dilation": 1,
                "filter_dilation": 1,
                "data_format": "channel_first",
                "precision": "high",
                "kernel_initializer": nn.initializers.constant(1, jnp.float16),
                "dtype": jnp.float32
            },
            jnp.ones((4, 3, 32, 32), jnp.bfloat16),
            jnp.full((4, 16, 28, 28), 75, jnp.bfloat16),
            jnp.ones((16, 3, 5, 5), jnp.float32)
        ),
        (
            {
                "rng": random.PRNGKey(1),
                "out_channels": 16,
                "filter_shape": 5,
                "padding": (0, (0, 0)),
                "input_dilation": (1, 1),
                "filter_dilation": (1, 1),
                "data_format": "channel_last",
                "accum_dtype": jnp.float32,
                "kernel_initializer": nn.initializers.constant(1, jnp.float16),
                "dtype": jnp.float32
            },
            jnp.ones((4, 32, 32, 3), jnp.bfloat16),
            jnp.full((4, 28, 28, 16), 75, jnp.float32),
            jnp.ones((16, 5, 5, 3), jnp.float32)
        ),
        (
            {
                "rng": random.PRNGKey(2),
                "out_channels": 16,
                "filter_shape": 3,
                "data_format": ("HWDC", "HWDOI", "CHWD"),
                "kernel_initializer": nn.initializers.constant(2, jnp.float16),
                "dtype": jnp.bfloat16
            },
            jnp.ones((4, 32, 32, 32, 3), jnp.float32),
            jnp.full((4, 16, 30, 30, 30), 162, jnp.float32),
            jnp.full((3, 3, 3, 16, 3), 2, jnp.bfloat16)
        ),
        (
            {
                "rng": random.PRNGKey(3),
                "out_channels": 16,
                "filter_shape": 3,
                "precision": ("high", "high"),
                "accum_dtype": jnp.float32,
                "kernel_initializer": nn.initializers.constant(3, jnp.float16),
                "dtype": jnp.bfloat16
            },
            jnp.ones((4, 32, 3), jnp.bfloat16),
            jnp.full((4, 30, 16), 27, jnp.float32),
            jnp.full((16, 3, 3), 3, jnp.bfloat16)
        ),
    ]
)
def test_conv(config, x, expected_output, expected_conv_kernel):
    layer, (t_acts, new_t_layer), (i_acts, new_i_layer) = layer_test_results(
        Conv, config, x
    )
    assert_equal_array(layer.conv_kernel.data, expected_conv_kernel)

    assert_equal_array(t_acts, expected_output)
    assert_equal_array(new_t_layer.conv_kernel.data, expected_conv_kernel)

    assert_equal_array(i_acts, expected_output)
    assert_equal_array(new_i_layer.conv_kernel.data, expected_conv_kernel)
