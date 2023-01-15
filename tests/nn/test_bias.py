import jax
from jax import (
    numpy as jnp,
    random,
    nn,
    lax
)
from mlax.nn import Bias
import pytest

@pytest.mark.parametrize(
    "config,input,expected_output,expected_bias_weight",
    [
        (
            {
                "rng": random.PRNGKey(0),
                "in_features": (None, 1, -1),
                "bias_initializer": nn.initializers.constant(1, jnp.float16),
                "dtype": jnp.float32
            },
            random.normal(random.PRNGKey(1), (2, 4, 4, 3), jnp.bfloat16),
            random.normal(random.PRNGKey(1), (2, 4, 4, 3), jnp.bfloat16) + 1,
            jnp.ones((1, 3), jnp.float32)
        ),
        (
            {
                "rng": random.PRNGKey(0),
                "in_features": (),
                "bias_initializer": nn.initializers.constant(2, jnp.float16),
                "dtype": jnp.float32
            },
            random.normal(random.PRNGKey(1), (2, 4, 4, 3), jnp.bfloat16),
            random.normal(random.PRNGKey(1), (2, 4, 4, 3), jnp.bfloat16) + 2,
            jnp.full((), 2, jnp.float32)
        ),
        (
            {
                "rng": random.PRNGKey(0),
                "in_features": -1,
                "bias_initializer": nn.initializers.constant(3, jnp.float16),
                "dtype": jnp.float32
            },
            random.normal(random.PRNGKey(1), (2, 8), jnp.bfloat16),
            random.normal(random.PRNGKey(1), (2, 8), jnp.bfloat16) + 3,
            jnp.full((8,), 3, jnp.float32)
        ),
    ]
)
def test_bias(
    config,
    input,
    expected_output,
    expected_bias_weight
):
    bias = Bias(
        **config
    )
    assert bias.bias_weight.data is None

    fwd = jax.jit(
        jax.vmap(
            Bias.fwd,
            in_axes = (None, None, 0, None, None),
            out_axes = (0, None)
        ),
        static_argnames="inference_mode"
    )

    activations, bias = fwd(
        bias,
        bias.trainables,
        input,
        None, # rng
        False # inference_mode
    )
    assert lax.eq(
        activations,
        expected_output
    ).all()

    assert lax.eq(
        bias.bias_weight.data,
        expected_bias_weight
    ).all()