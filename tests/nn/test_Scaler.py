from mlax.nn import Scaler
from common import assert_valid_pytree
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn,
    jit
)

dtype = jnp.float32
op_dtype = jnp.float16
key1, key2 = random.split(random.PRNGKey(0), 2)
inputs = random.normal(key1, (2, 4, 4, 3), dtype=op_dtype)
trainables, non_trainables, hyperparams = Scaler.init(
    key2,
    in_feature_shape=(None, 1, 3),
    scaler_initializer=nn.initializers.constant(2, jnp.bfloat16),
    dtype=dtype,
)

def test_init():
    assert_valid_pytree(trainables, non_trainables, hyperparams)
    assert lax.eq(
        trainables,
        jnp.full((1, 3), 2, dtype)
    ).all()
    assert non_trainables is None

def test_fwd():
    activations, new_ntr =  jit(Scaler.fwd, static_argnames="hyperparams")(
        inputs, trainables, non_trainables, hyperparams
    )
    assert lax.eq(
        activations,
        lax.convert_element_type(2 * inputs, op_dtype)
    ).all()
    assert new_ntr is None
