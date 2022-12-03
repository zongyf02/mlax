from mlax.nn import BatchNorm
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn,
    jit
)

key1, key2 = random.split(random.PRNGKey(0))
dtype = jnp.bfloat16
param_dtype = jnp.float32
inputs = random.normal(key1, (4, 16, 16, 3), dtype)
trainables, non_trainables, hyperparams = BatchNorm.init(
    key2,
    in_channels=3,
    channel_axis=2, # channel last
    dtype=dtype,
    param_dtype=param_dtype
)

def test_init():
    moving_mean, moving_variance = non_trainables
    assert lax.eq(
        moving_mean,
        jnp.ones((3,), param_dtype)
    ).all()
    assert lax.eq(
        moving_variance,
        jnp.zeros((3,), param_dtype)
    ).all()
    assert trainables is None

def test_fwd():
    fwd_fn = jit(
        BatchNorm.fwd,
        static_argnames=["hyperparams", "inference_mode"]
    )
    activations, (moving_mean, moving_var) = fwd_fn(
        inputs, trainables, non_trainables, hyperparams
    )
    assert lax.eq(
        activations,
        nn.standardize(inputs, (0, 1, 2))
    ).all()
    
    assert lax.eq(
        moving_mean,
        jnp.full((3,), 0.9, dtype) + inputs.mean((0, 1, 2)) * 0.1
    ).all()
    # Small error is tolerated because numpy variance is calculated differently
    assert ((moving_var - inputs.var((0, 1, 2)) * 0.1) < 1e-3).all()
