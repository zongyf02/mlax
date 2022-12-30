from mlax.functional import layer_norm, instance_norm, group_norm
import jax.numpy as jnp
from jax import (
    lax,
    random,
    nn
)

dtype = jnp.float32
key1, key2 = random.split(random.PRNGKey(0))
inputs1 = random.normal(key1, (2, 4, 8), dtype)
inputs2 = random.normal(key2, (2, 3, 4, 4, 4), dtype)

def test_layer_norm():
    activations = layer_norm(inputs1)
    assert lax.eq(
        activations,
        nn.standardize(inputs1, (1, 2))
    ).all()

    activations = layer_norm(inputs2)
    assert lax.eq(
        activations,
        nn.standardize(inputs2, (1, 2, 3, 4))
    ).all()

def test_instance_norm():
    activations = instance_norm(inputs1, channel_last=True)
    assert lax.eq(
        activations,
        nn.standardize(inputs1, (1,))
    ).all()

    activations = instance_norm(inputs2 , channel_last=False)
    assert lax.eq(
        activations,
        nn.standardize(inputs2, (2, 3, 4))
    ).all()

def test_group_norm():
    activations = group_norm(inputs1, 8, channel_last=True)
    assert lax.eq(
        activations,
        nn.standardize(inputs1, (1,))
    ).all()

    activations = group_norm(inputs2, 3, channel_last=False)
    assert lax.eq(
        activations,
        nn.standardize(inputs2, (2, 3, 4))
    ).all()
