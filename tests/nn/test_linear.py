import jax
from jax import (
    numpy as jnp,
    random,
    nn,
    lax
)
from mlax.nn import Linear
import pytest

@pytest.mark.parametrize(
    "config,input,expected_output,expected_kernel_weight",
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
                "rng": random.PRNGKey(0),
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
                "rng": random.PRNGKey(1),
                "out_features": 4,
                "precision": "float32",
                "transposed_kernel": True,
                "kernel_initializer": nn.initializers.constant(2, jnp.float16),
                "accum_dtype": jnp.float32,
                "dtype": jnp.float32
            },
            jnp.ones((3, 4), jnp.bfloat16),
            jnp.full((3, 4), 8, jnp.float32),
            jnp.full((4, 4), 2, jnp.float32)
        ),
    ]
)
def test_linear(
    config,
    input,
    expected_output,
    expected_kernel_weight
):
    linear = Linear(
        **config
    )
    assert linear.kernel_weight.data is None

    fwd = jax.jit(
        jax.vmap(
            Linear.fwd,
            in_axes = (None, None, 0, None, None),
            out_axes = (0, None)
        ),
        static_argnames="inference_mode"
    )

    activations, linear = fwd(
        linear,
        linear.trainables,
        input,
        None, # rng
        False # inference_mode
    )
    assert lax.eq(
        activations,
        expected_output
    ).all()

    assert lax.eq(
        linear.kernel_weight.data,
        expected_kernel_weight
    ).all()