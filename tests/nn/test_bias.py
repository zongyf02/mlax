from mlax.nn import bias
import jax.numpy as jnp
from jax import (
    nn,
    lax,
    random
)

key1, key2 = random.split(random.PRNGKey(0))

shape1 = (2, 4, 3)
type1 = "bfloat16"
inputs1 = jnp.zeros(shape1, dtype=type1)
weights1 = bias.init(
    key1,
    shape1[1:],
    bias_initializer= nn.initializers.constant(1, dtype="float64"),
    dtype = type1 # Should override bias initializer's dtype
)

shape2 = (7, 9)
type2 = "float16"
inputs2 = jnp.zeros(shape2, dtype=type2)
weights2 = bias.init(
    key2,
    shape2,
    bias_initializer= nn.initializers.constant(1, dtype="float32"),
    dtype = type2 # Should override bias initializer's dtype
)

def test_init():
    assert lax.eq(
        weights1,
        jnp.ones(shape1[1:], dtype=type1)
    ).all()
    
    assert lax.eq(
        weights2,
        jnp.ones(shape2, dtype=type2)
    ).all()

def test_fwd():
    activations = bias.fwd(
        inputs1, weights1
    )
    assert lax.eq(
        activations,
        jnp.ones(shape1, dtype=type1)
    ).all()

    activations = bias.fwd(
        inputs2, weights2
    )
    assert lax.eq(
        activations,
        jnp.ones(shape2, dtype=type2)
    ).all()
