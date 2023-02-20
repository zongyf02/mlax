from mlax.nn.functional import dropout
import jax.numpy as jnp
from jax import (
    lax,
    random
)
import pytest

@pytest.mark.parametrize(
    "input,params,expected_output",
    [
        (
            random.normal(random.PRNGKey(0), (2, 4, 3), jnp.float16),
            {
                "rng": random.PRNGKey(1),
                "rate": 0.0
            },
            random.normal(random.PRNGKey(0), (2, 4, 3), jnp.float16)
        )
    ]
)
def test_dropout(input, params, expected_output):
    activations = dropout(input, **params)
    assert lax.eq(
        activations,
        expected_output
    ).all()
