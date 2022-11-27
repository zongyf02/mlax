from mlax.nn import Bias
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn,
    jit,
    value_and_grad
)

dtype = jnp.float16
inputs = jnp.zeros((2, 4, 3), dtype=dtype)
trainables, non_trainables, hyperparams = Bias.init(
    random.PRNGKey(0),
    in_feature_shape=(4, 3),
    bias_dims=(1,),
    bias_initializer=nn.initializers.constant(1, dtype=jnp.float32),
    dtype=dtype # Should override bias initializer's dtype
)

def test_init():
    assert lax.eq(
        trainables,
        jnp.ones((3,), dtype=dtype)
    ).all()
    assert non_trainables is None

def test_fwd():
    activations, new_ntr =  jit(Bias.fwd, static_argnames="hyperparams")(
        inputs, trainables, non_trainables, hyperparams
    )
    assert lax.eq(
        activations,
        jnp.ones((2, 4, 3), dtype=dtype)
    ).all()
    assert new_ntr == non_trainables
