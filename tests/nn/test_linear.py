from mlax.nn import linear
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn
)

in_type = jnp.bfloat16
out_type = jnp.float32
inputs = jnp.ones((2, 4), dtype=in_type)
weights = linear.init(
    random.PRNGKey(0),
    in_features=4, out_features=3,
    kernel_initializer=nn.initializers.constant(1, dtype=out_type),
    dtype=in_type # Should override kernel initializer's dtype
)

def test_init():
    assert lax.eq(
        weights,
        jnp.ones((4, 3), dtype=in_type)
    ).all()

def test_fwd():
    activations = linear.fwd(
        inputs, weights,
        preferred_element_type=out_type # accumulation type
    )
    assert lax.eq(
        activations,
        jnp.full((2, 3), 4, dtype=out_type)
    ).all()
