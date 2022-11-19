from mlax.nn import bias
import jax.numpy as jnp
from jax import (
    nn,
    lax,
    random
)


shape = (2, 4, 3)
dtype = jnp.float16
inputs = jnp.zeros(shape, dtype=dtype)
weights = bias.init(
    random.PRNGKey(0),
    shape[1:],
    bias_initializer= nn.initializers.constant(1, dtype=jnp.float32),
    dtype = dtype # Should override bias initializer's dtype
)

def test_init():
    assert lax.eq(
        weights,
        jnp.ones(shape[1:], dtype=dtype)
    ).all()

def test_fwd():
    activations = bias.fwd(
        inputs, weights
    )
    assert lax.eq(
        activations,
        jnp.ones(shape, dtype=dtype)
    ).all()
