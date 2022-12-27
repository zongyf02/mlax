from mlax.functional import dropout
from common import assert_valid_pytree
from mlax.nn import F_rng
import jax.numpy as jnp
from jax import (
    lax,
    random,
    jit
)

dtype = jnp.float16
key1, key2 = random.split(random.PRNGKey(0))
inputs = random.normal(key1, (2, 4, 3), dtype)
trainables, non_trainables, hyperparams = F_rng.init(
    lambda x, key, infer: dropout(x, key, 1.0, infer)
)

def test_init():
    assert_valid_pytree(trainables, non_trainables, hyperparams)
    assert trainables is None
    assert non_trainables is None

def test_fwd():
    activations, new_ntr = jit(F_rng.fwd, static_argnames="hyperparams")(
        inputs, trainables, non_trainables, key2, hyperparams
    )
    assert lax.eq(
        activations,
        inputs
    ).all()
    assert new_ntr is None
