from mlax.blocks import Linear
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn
)

inputs = jnp.ones((4,), dtype="bfloat16")
weights = Linear.init(
    random.PRNGKey(0),
    in_features=4, out_features=3,
    kernel_initializer=nn.initializers.constant(1, dtype="float64"),
    bias_initializer=nn.initializers.ones,
    dtype="bfloat16" # Should override kernel initializer's dtype
)

def test_init():
    assert lax.eq(
        weights.kernel,
        jnp.ones((4, 3), dtype="bfloat16")
    ).all()
    assert lax.eq(
        weights.bias,
        jnp.ones((3,), dtype="bfloat16")
    ).all()

def test_fwd():
    activations = Linear.fwd(
        inputs, weights 
    )
    assert lax.eq(
        activations,
        jnp.full((3,), 5, dtype="bfloat16")
    ).all()
