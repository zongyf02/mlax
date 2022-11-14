from mlax.nn import dropout
from jax import (
    lax,
    random
)

type = "bfloat16"
key1, key2 = random.split(random.PRNGKey(0))

inputs = random.normal(key1, (2, 4, 3), type)

def test_dropout():
    activations = dropout(inputs, key2, 1.0, train=True)
    assert lax.eq(
        activations,
        inputs
    ).all()
