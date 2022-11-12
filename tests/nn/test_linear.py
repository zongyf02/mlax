from mlax.nn import linear
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn
)


inputs = jnp.ones((2, 4), dtype="bfloat16")
weights = linear.init(
    random.PRNGKey(0),
    in_features=4, out_features=3,
    kernel_initializer=nn.initializers.constant(1, dtype="float64"),
    dtype="bfloat16" # Should override kernel initializer's dtype
)

def test_init():
    assert lax.eq(
        weights,
        jnp.ones((4, 3), dtype="bfloat16")
    ).all()

def test_fwd():
    activations = linear.fwd(
        inputs, weights,
        preferred_element_type="float32" # accumulation type
    )
    assert lax.eq(
        activations,
        jnp.full((2, 3), 4, dtype="float32")
    ).all()
