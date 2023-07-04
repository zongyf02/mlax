import pytest
from jax import (
    numpy as jnp,
    random
)
from mlax.nn.functional import dropout
from mlax._test_utils import assert_equal_array

@pytest.mark.parametrize(
    "input,params,expected_output",
    [
        (
            random.normal(random.PRNGKey(0), (2, 4, 3), jnp.float16),
            {
                "rng": random.PRNGKey(1),
                "rate": 0.0,
                "axis": (0, 1, 2)
            },
            random.normal(random.PRNGKey(0), (2, 4, 3), jnp.float16)
        ),
        (
            random.normal(random.PRNGKey(0), (2, 4, 3), jnp.float16),
            {
                "rng": random.PRNGKey(1),
                "rate": 0.0,
                "axis": 0
            },
            random.normal(random.PRNGKey(0), (2, 4, 3), jnp.float16)
        )
    ]
)
def test_dropout(input, params, expected_output):
    activations = dropout(input, **params)
    assert_equal_array(activations, expected_output)