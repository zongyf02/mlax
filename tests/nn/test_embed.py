import jax
from jax import (
    numpy as jnp,
    random,
    nn,
    lax
)
from mlax import (
    fwd,
    is_trainable
)
from mlax.nn import Embed
import pytest

def range_initializer(key, shape, dtype):
    assert len(shape) == 2
    assert shape[1] == 1
    return jnp.expand_dims(jnp.arange(shape[0], dtype=dtype), axis=1)

@pytest.mark.parametrize(
    "config,expected_embed_weight,input,expected_output",
    [
        (
            {
                "rng": random.PRNGKey(0),
                "vocab_size": 10,
                "embed_dim": 8,
                "embed_initializer": nn.initializers.ones,
                "dtype": jnp.float32
            },
            jnp.ones((10, 8), dtype=jnp.float32),
            jnp.arange(11),
            jnp.concatenate(
                [
                    jnp.ones((10, 8), jnp.float32),
                    jnp.full((1, 8), jnp.nan, jnp.float32)
                ]
            )
        ),
        (
            {
                "rng": random.PRNGKey(1),
                "vocab_size": 8,
                "embed_dim": 12,
                "embed_initializer": nn.initializers.zeros,
                "dtype": jnp.int8
            },
            jnp.zeros((8, 12), jnp.int8),
            jnp.arange(9, dtype=jnp.int8),
            jnp.concatenate(
                [
                    jnp.zeros((8, 12), jnp.int8),
                    jnp.full((1, 12), -jnp.inf, jnp.int8)
                ]
            )
        ),
        (
            {
                "rng": random.PRNGKey(2),
                "vocab_size": 16,
                "embed_dim": 1,
                "embed_initializer": range_initializer,
                "dtype": jnp.float16
            },
            range_initializer(random.PRNGKey(2), (16, 1), jnp.float16),
            jnp.arange(17, dtype=jnp.int16),
            jnp.expand_dims(
                jnp.concatenate(
                    [
                        jnp.arange(16, dtype=jnp.float16),
                        jnp.full(1, jnp.nan, jnp.float16)
                    ]
                ),
                axis=1
            )
        )
    ]
)
def test_embed(
    config,
    expected_embed_weight,
    input,
    expected_output
):
    embed = Embed(
        **config
    )
    assert lax.eq(
        embed.embed_weight.data,
        expected_embed_weight
    ).all()

    fwd_jit = jax.jit(
        jax.vmap(
            fwd,
            in_axes = (None, None, 0, None, None),
            out_axes = (0, None)
        ),
        static_argnames="inference_mode"
    )

    activations, embed = fwd_jit(
        *embed.partition(),
        input,
        None, # rng
        False # inference_mode
    )
    assert jnp.logical_or(
        jnp.logical_and(
            jnp.isnan(activations),
            jnp.isnan(expected_output)
        ), # NaN are not equal to NaN
        lax.eq(
            activations,
            expected_output
        )
    ).all()
