from mlax.nn.functional import (
    dot_product_attention_logits,
    apply_attention_mask,
    apply_attention_weights
)
import jax.numpy as jnp
from jax import (
    numpy as jnp,
    lax,
    nn
)
import pytest

@pytest.mark.parametrize(
    "query,key,value,mask,expected_logits,expected_weights,expected_activations",
    [
        (
            jnp.ones((4, 8, 16), jnp.float16), # query
            jnp.ones((4, 8, 16), jnp.float16), # key
            jnp.full((4, 8, 16), 4, jnp.float16), # value
            jnp.concatenate((
                jnp.ones((8, 4, 2), bool),
                jnp.zeros((8, 4, 2), bool)
            ), axis=-1), # Mask
            jnp.full((8, 4, 4), 4, jnp.float16), # expected logits
            jnp.concatenate((
                jnp.full((8, 4, 2), 0.5, jnp.float16),
                jnp.zeros((8, 4, 2), jnp.float16)
            ), axis=-1), # expected weights
            jnp.full((4, 8, 16), 4, jnp.float16)
        )
    ]
)
def test_dot_product_attention(
    query, key, value,
    mask,
    expected_logits, expected_weights, expected_activations
):
    logits = dot_product_attention_logits(
        query, key
    )
    assert lax.eq(
        logits,
        expected_logits
    ).all()

    weights = nn.softmax(
        apply_attention_mask(
            logits, mask
        )
    )
    assert lax.eq(
        weights,
        expected_weights
    ).all()

    activations = apply_attention_weights(
        value, weights
    )
    assert lax.eq(
        activations,
        expected_activations
    ).all()
