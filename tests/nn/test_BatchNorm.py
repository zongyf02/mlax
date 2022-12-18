from mlax.nn import BatchNorm
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn,
    jit
)

key1, key2 = random.split(random.PRNGKey(0))
dtype = jnp.float16
op_dtype = jnp.float32
inputs = random.normal(key1, (4, 16, 16, 3), op_dtype)
trainables, non_trainables, hyperparams = BatchNorm.init(
    key2,
    in_channels=3,
    channel_axis=2, # channel last
    dtype=dtype
)

def test_init():
    moving_mean, moving_variance = non_trainables
    assert lax.eq(
        moving_mean,
        jnp.zeros((3,), dtype)
    ).all()
    assert lax.eq(
        moving_variance,
        jnp.ones((3,), dtype)
    ).all()
    assert trainables is None

def test_fwd():
    fwd_fn = jit(
        BatchNorm.fwd,
        static_argnames=["hyperparams", "inference_mode"]
    )
    activations, (moving_mean, moving_var)  = fwd_fn(
        inputs, trainables, non_trainables, hyperparams, inference_mode=False
    )
    assert lax.eq(
        activations,
        nn.standardize(inputs, (0, 1, 2))
    ).all()
    
    assert lax.eq(
        moving_mean,
        inputs.mean((0, 1, 2), dtype=op_dtype).astype(dtype) * 0.1
    ).all()
    # Small error tolerated because numpy var is calculated differently
    assert (
        lax.abs(lax.sub(
            moving_var,
            (
                jnp.ones((3,), dtype=dtype) * 0.9 +
                inputs.var((0, 1, 2), dtype=op_dtype).astype(dtype) * 0.1
            )
        )) < 1e-6
    ).all()
    
    activations, _ = fwd_fn(
        inputs, trainables, non_trainables, hyperparams, inference_mode=True
    )
    assert (
       lax.abs(lax.sub(activations, inputs)) < 1e-4
    ).all()
