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
inputs = jnp.ones((2, 3, 4), op_dtype)
trainables1, non_trainables1, hyperparams1 = Linear.init(
    random.PRNGKey(0),
    in_features=4, out_features=3,
    precision=("float32", "float32"),
    transposed_kernel=True,
    kernel_initializer=nn.initializers.constant(1, jnp.float16),
    accum_dtype=dtype,
    dtype=dtype # Should override kernel initializer's dtype
)
trainables2, non_trainables2, hyperparams2 = Linear.init(
    random.PRNGKey(0),
    in_features=4, out_features=3,
    precision="float32",
    transposed_kernel=False,
    kernel_initializer=nn.initializers.constant(1, jnp.float16),
    dtype=dtype # Should override kernel initializer's dtype
)

def test_init():
    assert_valid_pytree(trainables1, non_trainables1, hyperparams1)
    assert_valid_pytree(trainables2, non_trainables2, hyperparams2)

    assert lax.eq(
        trainables1,
        jnp.ones((3, 4), dtype)
    ).all()
    assert non_trainables1 is None

    assert lax.eq(
        trainables2,
        jnp.ones((4, 3), dtype)
    ).all()
    assert non_trainables2 is None

def test_fwd():
    fwd = jit(Linear.fwd, static_argnames="hyperparams")

    activations, new_ntr = fwd(
        inputs, trainables1, non_trainables1, hyperparams1
    )
    assert lax.eq(
        activations,
        jnp.full((2, 3, 3), 4, dtype)
    ).all()
    assert new_ntr is None

    activations, new_ntr = fwd(
        inputs, trainables2, non_trainables2, hyperparams2
    )
    assert lax.eq(
        activations,
        jnp.full((2, 3, 3), 4, op_dtype)
    ).all()
    assert new_ntr is None 
