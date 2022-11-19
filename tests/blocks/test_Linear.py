from mlax.blocks import Linear
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn
)

in_type = jnp.float16
inputs = jnp.ones((3, 4), dtype=in_type)
weights = Linear.init(
    random.PRNGKey(0),
    in_features=4, out_features=3,
    kernel_initializer=nn.initializers.constant(1, dtype=jnp.float32),
    bias_initializer=nn.initializers.ones,
    dtype=in_type # Should override kernel initializer's dtype
)

def test_init():
    assert lax.eq(
        weights.kernel,
        jnp.ones((4, 3), dtype=in_type)
    ).all()
    assert lax.eq(
        weights.bias,
        jnp.ones((3,), dtype=in_type)
    ).all()

def test_fwd():
    activations = Linear.fwd(
        inputs, weights 
    )
    assert lax.eq(
        activations,
        jnp.full((3, 3), 5, dtype=in_type)
    ).all()
