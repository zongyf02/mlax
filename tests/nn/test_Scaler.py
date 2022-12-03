from mlax.nn import Scaler
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn,
    jit
)

dtype = jnp.float16
param_dtype = jnp.float32
inputs = jnp.full((2, 4, 3), 2, dtype=dtype)
trainables, non_trainables, hyperparams = Scaler.init(
    random.PRNGKey(0),
    in_feature_shape=(4, 3),
    scaler_dims=(1,),
    dtype=dtype,
    scaler_initializer=nn.initializers.constant(1, dtype),
    param_dtype=param_dtype # Should override bias initializer's dtype
)

def test_init():
    assert lax.eq(
        trainables,
        jnp.ones((3,), param_dtype)
    ).all()
    assert non_trainables is None

def test_fwd():
    activations, new_ntr =  jit(Scaler.fwd, static_argnames="hyperparams")(
        inputs, trainables, non_trainables, hyperparams
    )
    assert lax.eq(
        activations,
        jnp.full((2, 4, 3), 2, dtype)
    ).all()
    assert new_ntr is None
