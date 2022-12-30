from mlax.nn import F
from mlax.functional import pool
from common import assert_valid_pytree
import jax.numpy as jnp
from jax import (
    lax,
    jit
)

dtype = jnp.float16
inputs = jnp.full((2, 4, 4, 1), -1, dtype=dtype)
trainables, non_trainables, hyperparams = F.init(
    lambda x: pool(
        x, lax.convert_element_type(1, dtype), lax.mul, 2, 2, channel_last=True
    ),
    lambda x: pool(
        x, lax.convert_element_type(2, dtype), lax.mul, 2, 2, channel_last=True
    )
)

def test_init():
    assert_valid_pytree(trainables, non_trainables, hyperparams)
    assert trainables is None
    assert non_trainables is None

def test_fwd():
    fwd = jit(F.fwd, static_argnames=["hyperparams", "inference_mode"])
    activations, new_ntr = fwd(
        inputs, trainables, non_trainables, hyperparams, inference_mode=False
    )
    assert lax.eq(
        activations,
        jnp.ones((2, 3, 3, 1), dtype=dtype)
    ).all()
    assert new_ntr is None 

    activations, new_ntr = fwd(
        inputs, trainables, non_trainables, hyperparams, inference_mode=True
    )
    assert lax.eq(
        activations,
        jnp.full((2, 3, 3, 1), 2, dtype=dtype)
    ).all()
    assert new_ntr is None 
