import jax
from jax import (
    numpy as jnp,
    random,
    nn,
    lax
)
from mlax import (
    fwd,
    is_trainable
)
from mlax.nn import Conv
import pytest

@pytest.mark.parametrize(
    "config,input,expected_output,expected_kernel_weight",
    [
        (
            {
                "rng": random.PRNGKey(0),
                "n_spatial_dims": 2,
                "out_channels": 16,
                "filter_shape": (5, 5),
                "strides": 1,
                "padding": 0,
                "input_dilation": 1,
                "filter_dilation": 1,
                "channel_last": False,
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
                "n_spatial_dims": 2,
                "out_channels": 16,
                "filter_shape": 5,
                "padding": (0, (0, 0)),
                "input_dilation": (1, 1),
                "filter_dilation": (1, 1),
                "channel_last": True,
                "precision": None,
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
                "n_spatial_dims": 3,
                "out_channels": 16,
                "filter_shape": 3,
                "channel_last": True,
                "kernel_initializer": nn.initializers.constant(2, jnp.float16),
                "dtype": jnp.float32
            },
            jnp.ones((4, 32, 32, 32, 3), jnp.bfloat16),
            jnp.full((4, 30, 30, 30, 16), 162, jnp.bfloat16),
            jnp.full((16, 3, 3, 3, 3), 2, jnp.float32)
        ),
        (
            {
                "rng": random.PRNGKey(3),
                "n_spatial_dims": 1,
                "out_channels": 16,
                "filter_shape": 3,
                "accum_dtype": jnp.float32,
                "kernel_initializer": nn.initializers.constant(3, jnp.float16),
                "dtype": jnp.float32
            },
            jnp.ones((4, 3, 32), jnp.bfloat16),
            jnp.full((4, 16, 30), 27, jnp.float32),
            jnp.full((16, 3, 3), 3, jnp.float32)
        ),
    ]
)
def test_linear(
    config,
    input,
    expected_output,
    expected_kernel_weight
):
    conv = Conv(
        **config
    )
    assert conv.kernel_weight.data is None

    fwd_jit = jax.jit(
        jax.vmap(
            fwd,
            in_axes = (None, None, 0, None, None),
            out_axes = (0, None)
        ),
        static_argnames="inference_mode"
    )

    activations, conv = fwd_jit(
        *conv.partition(),
        input,
        None, # rng
        False # inference_mode
    )
    assert lax.eq(
        activations,
        expected_output
    ).all()

    assert lax.eq(
        conv.kernel_weight.data,
        expected_kernel_weight
    ).all()
