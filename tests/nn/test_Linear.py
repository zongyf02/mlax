from mlax.nn import Linear
from common import assert_valid_pytree
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn,
    jit
)

dtype = jnp.float32
op_dtype = jnp.bfloat16
inputs = jnp.ones((2, 4), op_dtype)
trainables, non_trainables, hyperparams = Linear.init(
    random.PRNGKey(0),
    in_features=4, out_features=3,
    precision=("float32", "float32"),
    transposed_kernel=True,
    kernel_initializer=nn.initializers.constant(1, jnp.float16),
    accum_dtype=dtype,
    dtype=dtype # Should override kernel initializer's dtype
)

def test_init():
    assert_valid_pytree(trainables, non_trainables, hyperparams)
    assert lax.eq(
        trainables,
        jnp.ones((3, 4), dtype)
    ).all()
    assert non_trainables is None

def test_fwd():
    activations, new_ntr = jit(Linear.fwd, static_argnames="hyperparams")(
        inputs, trainables, non_trainables, hyperparams
    )
    assert lax.eq(
        activations,
        jnp.full((2, 3), 4, dtype)
    ).all()
    assert new_ntr is None 
