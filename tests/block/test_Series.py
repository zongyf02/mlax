from mlax.nn import Linear, Bias, F_rng, F
from mlax.functional import dropout
from mlax.block import series_fwd, series_rng_fwd 
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn,
    jit
)

key1, key2, key3 = random.split(random.PRNGKey(0), 3)
in_dtype = jnp.bfloat16
out_dtype = jnp.float32
inputs = jnp.ones((2, 4), dtype=in_dtype)
linear_vars = Linear.init(
    key1,
    in_feature_shape=[4], out_feature_shape=[3],
    kernel_initializer=nn.initializers.ones,
    dtype=in_dtype,
    accum_dtype=out_dtype
)
bias_vars = Bias.init(
    key2,
    in_feature_shape=[3],
    bias_dims=[0],
    bias_initializer=nn.initializers.ones,
    dtype=out_dtype
)

def test_series_fwd():
    trainables, non_trainables, hyperparams = zip(
        linear_vars, bias_vars,
        F.init(
            lambda x: x,
            lambda _: 0
        )
    )
    fwd = jit(series_fwd, static_argnames=["hyperparams", "inference_mode"])
    
    activations, new_ntr = fwd(
        inputs, trainables, non_trainables, hyperparams, False
    )
    assert lax.eq(
        activations,
        jnp.full((2, 3), 5, dtype=out_dtype)
    ).all()
    assert non_trainables == new_ntr

    activations, new_ntr = fwd(
        inputs, trainables, non_trainables, hyperparams, True
    )
    assert activations == 0
    assert non_trainables == new_ntr

def test_series_rng_fwd():
    trainables, non_trainables, hyperparams = zip(
        linear_vars, bias_vars,
        F_rng.init(
            lambda x, key, infer: 0 if infer else dropout(x, key, 1.0, infer)
        )
    )
    fwd = jit(series_rng_fwd, static_argnames=["hyperparams", "inference_mode"])

    activations, new_ntr = fwd(
        inputs, trainables, non_trainables, key3, hyperparams, False
    )
    assert lax.eq(
        activations,
        jnp.full((2, 3), 5, dtype=out_dtype)
    ).all()
    assert non_trainables == new_ntr
    
    activations, new_ntr = fwd(
        inputs, trainables, non_trainables, key3, hyperparams, True
    )
    assert activations == 0
    assert non_trainables == new_ntr
