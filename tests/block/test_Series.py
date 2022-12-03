from mlax.nn import Conv, Linear, Bias, F_rng, F
from mlax.functional import dropout
from mlax.block import series_fwd, series_rng_fwd 
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn,
    jit
)

keys_iter = iter(random.split(random.PRNGKey(0), 5))
dtype = jnp.float32
inputs = jnp.ones((2, 3, 8, 8), dtype)
conv_vars = Conv.init(
    next(keys_iter),
    ndims=2,
    in_channels=3,
    out_channels=4,
    filter_shape=3,
    strides=1,
    kernel_initializer=nn.initializers.ones
)
flatten_vars = F.init(
    lambda x: jnp.reshape(x, (2, -1))
)
linear_vars = Linear.init(
    next(keys_iter),
    in_features=144, out_features=3,
    kernel_initializer=nn.initializers.ones
)
bias_vars = Bias.init(
    next(keys_iter),
    in_feature_shape=[3],
    bias_dims=[0],
    bias_initializer=nn.initializers.ones
)

def test_series_fwd():
    trainables, non_trainables, hyperparams = zip(
        conv_vars, flatten_vars,
        linear_vars, bias_vars,
        F.init(
            lambda x: x,
            lambda _: 0
        )
    )
    fwd = jit(series_fwd, static_argnames=["hyperparams", "inference_mode"])
    
    activations, _ = fwd(
        inputs, trainables, non_trainables, hyperparams, False
    )
    assert lax.eq(
        activations,
        jnp.full((2, 3), 3889, dtype)
    ).all()

    activations, _ = fwd(
        inputs, trainables, non_trainables, hyperparams, True
    )
    assert activations == 0

def test_series_rng_fwd():
    trainables, non_trainables, hyperparams = zip(
        conv_vars, flatten_vars,
        linear_vars, bias_vars,
        F_rng.init(
            lambda x, key, infer: 0 if infer else dropout(x, key, 1.0, infer)
        )
    )
    fwd = jit(series_rng_fwd, static_argnames=["hyperparams", "inference_mode"])

    activations, _ = fwd(
        inputs, trainables, non_trainables, next(keys_iter), hyperparams, False
    )
    assert lax.eq(
        activations,
        jnp.full((2, 3), 3889, dtype)
    ).all()
    
    activations, _ = fwd(
        inputs, trainables, non_trainables, next(keys_iter), hyperparams, True
    )
    assert activations == 0
