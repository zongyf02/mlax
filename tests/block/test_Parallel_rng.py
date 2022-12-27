from mlax.nn import Conv, Scaler, F_rng
from mlax.block import Parallel_rng
from mlax.functional import dropout
from common import assert_valid_pytree
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn,
    jit
)

keys_iter = iter(random.split(random.PRNGKey(0), 4))
dtype = jnp.float32
inputs = [
    jnp.ones((2, 8, 8, 3), dtype),
    jnp.ones((2, 4), dtype),
    jnp.ones((2, 3), dtype)
]
trainables, non_trainables, hyperparams = Parallel_rng.init(
    Conv.init(
        next(keys_iter),
        ndims=2,
        in_channels=3,
        out_channels=4,
        filter_shape=3,
        channel_last=True,
        kernel_initializer=nn.initializers.ones
    ),
    Scaler.init(
        next(keys_iter),
        in_feature_shape=(4,)
    ),
    F_rng.init(
        lambda x, key, infer: 0 if infer else dropout(x, key, 1.0, infer)
    )
)

def test_init():
    assert_valid_pytree(trainables, non_trainables, hyperparams)

def test_fwd():
    fwd = jit(Parallel_rng.fwd, static_argnames=["hyperparams", "inference_mode"])
    conv_acts_ref = jnp.full((2, 6, 6, 4), 27, dtype)
    scaler_acts_ref = jnp.ones((2, 4), dtype)

    activations, ntr = fwd(
        inputs, trainables, non_trainables, next(keys_iter), hyperparams, False
    )
    assert lax.eq(
        activations[0],
        conv_acts_ref
    ).all()
    assert lax.eq(
        activations[1],
        scaler_acts_ref
    ).all()
    assert lax.eq(
        activations[2],
        inputs[2]
    ).all()
    non_trainables.__class__ == ntr.__class__

    activations, ntr = fwd(
        inputs, trainables, non_trainables, next(keys_iter), hyperparams, True
    )
    assert lax.eq(
        activations[0],
        conv_acts_ref
    ).all()
    assert lax.eq(
        activations[1],
        scaler_acts_ref
    ).all()
    assert activations[2] == 0
    non_trainables.__class__ == ntr.__class__
