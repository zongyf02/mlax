from mlax.functional import dropout
import jax.numpy as jnp
from jax import (
    lax,
    random
)

dtype = jnp.float16
key1, key2 = random.split(random.PRNGKey(0))
inputs = random.normal(key1, (2, 4, 3), dtype)

def test_dropout():
    activations = dropout(inputs, key2, 1.0, inference_mode=False)
    assert lax.eq(
        activations,
        inputs
    ).all()
