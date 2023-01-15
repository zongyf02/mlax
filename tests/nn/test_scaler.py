import jax
from jax import (
    numpy as jnp,
    random,
    nn,
    lax
)
from mlax.nn import Scaler
import pytest

@pytest.mark.parametrize(
    "config,input,expected_output,expected_scaler_weight",
    [
        (
            {
                "rng": random.PRNGKey(0),
                "in_features": (None, 1, -1),
                "scaler_initializer": nn.initializers.constant(1, jnp.float16),
                "dtype": jnp.float32
            },
            random.normal(random.PRNGKey(1), (2, 4, 4, 3), jnp.bfloat16),
            random.normal(random.PRNGKey(1), (2, 4, 4, 3), jnp.bfloat16),
            jnp.ones((1, 3), jnp.float32)
        ),
        (
            {
                "rng": random.PRNGKey(0),
                "in_features": (),
                "scaler_initializer": nn.initializers.constant(2, jnp.float16),
                "dtype": jnp.float32
            },
            random.normal(random.PRNGKey(1), (2, 4, 4, 3), jnp.bfloat16),
            random.normal(random.PRNGKey(1), (2, 4, 4, 3), jnp.bfloat16) * 2,
            jnp.full((), 2, jnp.float32)
        ),
        (
            {
                "rng": random.PRNGKey(0),
                "in_features": -1,
                "scaler_initializer": nn.initializers.constant(3, jnp.float16),
                "dtype": jnp.float32
            },
            random.normal(random.PRNGKey(1), (2, 8), jnp.bfloat16),
            random.normal(random.PRNGKey(1), (2, 8), jnp.bfloat16) * 3,
            jnp.full((8,), 3, jnp.float32)
        ),
    ]
)
def test_scaling(
    config,
    input,
    expected_output,
    expected_scaler_weight
):
    scaling = Scaler(
        **config
    )
    assert scaling.scaler_weight.data is None

    fwd = jax.jit(
        jax.vmap(
            Scaler.fwd,
            in_axes = (None, None, 0, None, None),
            out_axes = (0, None)
        ),
        static_argnames="inference_mode"
    )

    activations, scaling = fwd(
        scaling,
        scaling.trainables,
        input,
        None, # rng
        False # inference_mode
    )
    assert lax.eq(
        activations,
        expected_output
    ).all()

    assert lax.eq(
        scaling.scaler_weight.data,
        expected_scaler_weight
    ).all()