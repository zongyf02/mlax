from mlax.nn import BatchNorm, Conv, Linear, Bias, F_rng, F
from mlax.functional import dropout
from mlax.block import Series_rng
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn,
    jit
)

keys_iter = iter(random.split(random.PRNGKey(0), 6))
dtype = jnp.float32
inputs = jnp.ones((2, 3, 8, 8), dtype)
trainables, non_trainables, hyperparams = Series_rng.init(
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
    Series_rng.init(
        Linear.init(
            next(keys_iter),
            in_features=144, out_features=3,
            kernel_initializer=nn.initializers.ones
        )
    ),
    Series_rng.init(
        F_rng.init(
            lambda x, key, infer: 0 if infer else dropout(x, key, 1.0, infer)
        )
    )
)

def test_fwd():
    fwd = jit(Series_rng.fwd, static_argnames=["hyperparams", "inference_mode"])
    
    activations, _ = fwd(
        inputs, trainables, non_trainables, next(keys_iter), hyperparams, False
    )
    assert lax.eq(
        activations,
        jnp.full((2, 3), 3888, dtype)
    ).all()

    activations, _ = fwd(
        inputs, trainables, non_trainables, next(keys_iter), hyperparams, True
    )
    assert activations == 0
