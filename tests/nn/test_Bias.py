from mlax.nn import Bias
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn,
    jit
)

dtype = jnp.float16
key1, key2 = random.split(random.PRNGKey(0), 2)
inputs = random.normal(key1, (2, 4, 4, 3), dtype=dtype)
trainables, non_trainables, hyperparams = Bias.init(
    key2,
    in_feature_shape=(None, 1, 3),
    bias_initializer=nn.initializers.constant(1, jnp.float32),
    dtype=dtype # Should override bias initializer's dtype
)

def test_init():
    assert lax.eq(
        trainables,
        jnp.ones((1, 3), dtype)
    ).all()
    assert non_trainables is None

def test_fwd():
    activations, new_ntr =  jit(Bias.fwd, static_argnames="hyperparams")(
        inputs, trainables, non_trainables, hyperparams
    )
    assert lax.eq(
        activations,
        lax.convert_element_type(inputs + 1, dtype)
    ).all()
    assert new_ntr is None 
