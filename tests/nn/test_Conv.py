from mlax.nn import Conv
from common import assert_valid_pytree
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn,
    jit
)

key1, key2 = random.split(random.PRNGKey(0))
dtype=jnp.float32
op_dtype = jnp.float16
inputs1 = jnp.ones((4, 3, 32, 32), op_dtype)
inputs2 = jnp.ones((4, 32, 32, 3), op_dtype)
trainables1, non_trainables1, hyperparams1 = Conv.init(
    key1,
    ndims=2,
    in_channels=3,
    out_channels=16,
    filter_shape=(5, 5),
    padding=0,
    input_dilation=1,
    filter_dilation=1,
    precision="high",
    kernel_initializer=nn.initializers.constant(1, dtype=jnp.bfloat16),
    dtype=dtype
)
trainables2, non_trainables2, hyperparams2 = Conv.init(
    key2,
    ndims=2,
    in_channels=3,
    out_channels=16,
    filter_shape=5,
    strides=(1, 1),
    padding=(0, (0, 0)),
    input_dilation=(1, 1),
    filter_dilation=(1, 1),
    channel_last=True,
    precision=None,
    accum_dtype=dtype,
    kernel_initializer=nn.initializers.constant(1, dtype=jnp.bfloat16),
    dtype=dtype
)

def test_init():
    assert_valid_pytree(trainables1, non_trainables1, hyperparams1)
    assert_valid_pytree(trainables2, non_trainables2, hyperparams2)

    assert lax.eq(
        trainables1,
        jnp.ones((16, 3, 5, 5), dtype)
    ).all()
    assert non_trainables1 is None

    assert lax.eq(
        trainables2,
        jnp.ones((16, 5, 5, 3), dtype)
    ).all()
    assert non_trainables2 is None

def test_fwd():
    activations, new_ntr = jit(Conv.fwd, static_argnames="hyperparams")(
        inputs1,
        trainables1,
        non_trainables1,
        hyperparams1
    )
    assert lax.eq(
        activations,
        jnp.full((4, 16, 28, 28), 75, op_dtype)
    ).all()
    assert new_ntr is None 

    activations, new_ntr = Conv.fwd(
        inputs2,
        trainables2,
        non_trainables2,
        hyperparams2
    )
    assert lax.eq(
        activations,
        jnp.full((4, 28, 28, 16), 75, dtype)
    ).all()
    assert new_ntr is None
