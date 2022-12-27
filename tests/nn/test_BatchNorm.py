from mlax.nn import BatchNorm
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn,
    jit
)

key1, key2, key3, key4 = random.split(random.PRNGKey(0), 4)
dtype = jnp.float16
op_dtype = jnp.float32
inputs1 = random.normal(key1, (4, 16, 16, 3), op_dtype)
inputs2 = random.normal(key2, (4, 3, 16, 16), op_dtype)
trainables1, non_trainables1, hyperparams1 = BatchNorm.init(
    key3,
    in_channels=3,
    channel_last=True,
    dtype=dtype
)
trainables2, non_trainables2, hyperparams2 = BatchNorm.init(
    key4,
    in_channels=3,
    dtype=dtype
)

def test_init():
    moving_mean, moving_variance = non_trainables1
    assert lax.eq(
        moving_mean,
        jnp.zeros((3,), dtype)
    ).all()
    assert lax.eq(
        moving_variance,
        jnp.ones((3,), dtype)
    ).all()
    assert trainables1 is None

    moving_mean, moving_variance = non_trainables2
    assert lax.eq(
        moving_mean,
        jnp.zeros((3,), dtype)
    ).all()
    assert lax.eq(
        moving_variance,
        jnp.ones((3,), dtype)
    ).all()
    assert trainables2 is None

def test_fwd():
    fwd_fn = jit(
        BatchNorm.fwd,
        static_argnames=["hyperparams", "inference_mode"]
    )

    activations, (moving_mean, moving_var)  = fwd_fn(
        inputs1,
        trainables1, non_trainables1, hyperparams1,
        inference_mode=False
    )
    assert lax.eq(
        activations,
        nn.standardize(inputs1, (0, 1, 2))
    ).all()
    
    assert lax.eq(
        moving_mean,
        inputs1.mean((0, 1, 2), dtype=op_dtype).astype(dtype) * 0.1
    ).all()
    # Small error tolerated because numpy var is calculated differently
    assert (
        lax.abs(lax.sub(
            moving_var,
            (
                jnp.ones((3,), dtype=dtype) * 0.9 +
                inputs1.var((0, 1, 2), dtype=op_dtype).astype(dtype) * 0.1
            )
        )) < 1e-6
    ).all()

    activations, _ = fwd_fn(
        inputs1,
        trainables1, non_trainables1, hyperparams1,
        inference_mode=True
    )
    assert (
       lax.abs(lax.sub(activations, inputs1)) < 1e-4
    ).all()

    activations, (moving_mean, moving_var)  = fwd_fn(
        inputs2,
        trainables2, non_trainables2, hyperparams2,
        inference_mode=False
    )
    assert lax.eq(
        activations,
        nn.standardize(inputs2, (0, 2, 3))
    ).all()
    
    assert lax.eq(
        moving_mean,
        inputs2.mean((0, 2, 3), dtype=op_dtype).astype(dtype) * 0.1
    ).all()
    # Small error tolerated because numpy var is calculated differently
    assert (
        lax.abs(lax.sub(
            moving_var,
            (
                jnp.ones((3,), dtype=dtype) * 0.9 +
                inputs2.var((0, 2, 3), dtype=op_dtype).astype(dtype) * 0.1
            )
        )) < 1e-6
    ).all()
    
    activations, _ = fwd_fn(
        inputs2,
        trainables2, non_trainables2, hyperparams2,
        inference_mode=True
    )
    assert (
       lax.abs(lax.sub(activations, inputs2)) < 1e-4
    ).all()