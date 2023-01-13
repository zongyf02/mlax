from mlax.functional import (
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

dtype = jnp.float16
query = jnp.ones((2, 4, 8, 16), dtype)
key = jnp.ones((2, 4, 8, 16), dtype)
value = jnp.full((2, 4, 8, 16), 4, dtype)
logits = jnp.full((2, 8, 4, 4), 4, dtype)
mask = jnp.concatenate((
    jnp.ones((2, 8, 4, 2), bool),
    jnp.zeros((2, 8, 4, 2), bool)
), axis=-1)
weights = jnp.concatenate((
    jnp.full((2, 8, 4, 2), 0.5, dtype),
    jnp.zeros((2, 8, 4, 2), dtype)
), axis=-1)
activations = jnp.full((2, 4, 8, 16), 4, dtype)

def test_dot_product_attention_logits():
    _logits = dot_product_attention_logits(
        query, key, scaled=True
    )
    assert lax.eq(
        _logits,
        logits
    ).all()

def test_apply_attention_mask():
    _weights = nn.softmax(
        apply_attention_mask(
            logits, mask
        )
    )
    assert lax.eq(
        _weights,
        weights
    ).all()

def test_apply_attention_weights():
    _activations = apply_attention_weights(
        value, weights
    )
    assert lax.eq(
        _activations,
        activations
    ).all()