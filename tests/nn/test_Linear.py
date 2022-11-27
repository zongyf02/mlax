from mlax.nn import Linear
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn,
    jit
)

in_dtype = jnp.bfloat16
out_dtype = jnp.float32
inputs = jnp.ones((2, 4), dtype=in_dtype)
trainables, non_trainables, hyperparams = Linear.init(
    random.PRNGKey(0),
    in_feature_shape=(4,), out_feature_shape=(3,),
    kernel_initializer=nn.initializers.constant(1, dtype=jnp.float64),
    dtype=in_dtype, # Should override kernel initializer's dtype
    accum_dtype=out_dtype
)

def test_init():
    assert lax.eq(
        trainables,
        jnp.ones((3, 4), dtype=in_dtype)
    ).all()
    assert non_trainables is None

def test_fwd():
    activations, new_ntr = jit(Linear.fwd, static_argnames="hyperparams")(
        inputs, trainables, non_trainables, hyperparams
    )
    assert lax.eq(
        activations,
        jnp.full((2, 3), 4, dtype=out_dtype)
    ).all()
    assert new_ntr == non_trainables
