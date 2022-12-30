from mlax.nn import BatchNorm, Conv, Linear, Bias, F
from mlax.block import Series
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
inputs = jnp.ones((2, 3, 8, 8), dtype)
trainables, non_trainables, hyperparams = Series.init(
    BatchNorm.init(
        next(keys_iter),
        in_channels=3
    ),
    Bias.init(
        next(keys_iter),
        in_feature_shape=[3],
        bias_initializer=nn.initializers.ones
    ),
    Conv.init(
        next(keys_iter),
        ndims=2,
        in_channels=3,
        out_channels=4,
        filter_shape=3,
        strides=1,
        kernel_initializer=nn.initializers.ones
    ),
    F.init(
        lambda x: jnp.reshape(x, (2, -1))
    ),
    Series.init(
        Linear.init(
            next(keys_iter),
            in_features=144, out_features=3,
            kernel_initializer=nn.initializers.ones
        )
    ),
    F.init(
        lambda x: x,
        lambda _: 0
    )
)

def test_init():
    assert_valid_pytree(trainables, non_trainables, hyperparams)

def test_fwd():
    fwd = jit(Series.fwd, static_argnames=["hyperparams", "inference_mode"])
    
    activations, ntr = fwd(
        inputs, trainables, non_trainables, hyperparams, False
    )
    assert lax.eq(
        activations,
        jnp.full((2, 3), 3888, dtype)
    ).all()
    non_trainables.__class__ == ntr.__class__

    activations, ntr = fwd(
        inputs, trainables, non_trainables, hyperparams, True
    )
    assert activations == 0
    non_trainables.__class__ == ntr.__class__
