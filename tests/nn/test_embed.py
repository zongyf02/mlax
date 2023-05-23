from jax import (
    numpy as jnp,
    random,
    nn
)
import pytest
from mlax.nn import Embed
from mlax._test_utils import layer_test_results, assert_equal_array

def range_initializer(key, shape, dtype):
    assert len(shape) == 2
    assert shape[1] == 1
    return jnp.expand_dims(jnp.arange(shape[0], dtype=dtype), axis=1)

@pytest.mark.parametrize(
    "config,x,expected_embed_kernel,expected_output",
    [
        (
            {
                "rng": random.PRNGKey(0),
                "vocab_size": 10,
                "embed_dim": 8,
                "embed_initializer": nn.initializers.ones,
                "dtype": jnp.float32
            },
            jnp.arange(11),
            jnp.ones((10, 8), dtype=jnp.float32),
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
                "dtype": jnp.int16
            },
            jnp.arange(9, dtype=jnp.int8),
            jnp.zeros((8, 12), jnp.int16),
            jnp.concatenate(
                [
                    jnp.zeros((8, 12), jnp.int16),
                    jnp.full((1, 12), -jnp.inf, jnp.int16)
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
            jnp.arange(18, dtype=jnp.int16),
            jnp.expand_dims(jnp.arange(16, dtype=jnp.float16), axis=1),
            jnp.expand_dims(
                jnp.concatenate([
                    jnp.arange(16, dtype=jnp.float16),
                    jnp.full(2, jnp.nan, jnp.float16)
                ]),
                axis=1
            )
        )
    ]
)
def test_embed(config, x, expected_embed_kernel, expected_output):
    layer, (t_acts, new_t_layer), (i_acts, new_i_layer) = layer_test_results(
        Embed, config, x
    )
    assert_equal_array(layer.embed_kernel.data, expected_embed_kernel)

    assert_equal_array(t_acts, expected_output)
    assert_equal_array(new_t_layer.embed_kernel.data, expected_embed_kernel)

    assert_equal_array(i_acts, expected_output)
    assert_equal_array(new_i_layer.embed_kernel.data, expected_embed_kernel)
