from mlax.nn import bias
import jax.numpy as jnp
from jax import (
    nn,
    lax,
    random
)

shape = (4, 3)
inputs = jnp.zeros(shape, dtype="bfloat16")
weights = bias.init(
    random.PRNGKey(0),
    shape,
    bias_initializer= nn.initializers.constant(1, dtype="float64"),
    dtype = "bfloat16" # Should override kernel initializer's dtype
)

def test_init():
    assert lax.eq(
        weights,
        jnp.ones(shape, dtype="bfloat16")
    ).all()

def test_fwd():
    activations = bias.fwd(
        inputs, weights
    )
    assert lax.eq(
        activations,
        jnp.ones(shape, dtype="bfloat16")
    ).all()
