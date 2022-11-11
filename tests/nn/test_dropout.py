from mlax.nn import dropout
from jax import (
    lax,
    random
)

type = "bfloat16"
key1 = random.PRNGKey(0)
key1, key2 = random.split(key1)

inputs = random.normal(key1, (4, 3), type)

def test_fwd():
    activations = dropout.fwd(inputs, key2, 1.0, train=True)
    assert lax.eq(
        activations,
        inputs
    ).all()
