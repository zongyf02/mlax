from mlax.nn import conv
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn
)

in_type = jnp.float16
out_type = jnp.float32
inputs1 = jnp.ones((2, 3, 32, 32), dtype=in_type)
inputs2 = jnp.ones((2, 32, 32, 3), dtype=in_type)
weights = conv.init(
    random.PRNGKey(0),
    in_channels=3,
    out_channels=16,
    filter_shape=(3, 3),
    kernel_initializer=nn.initializers.constant(1, dtype=out_type),
    dtype=in_type
)

def test_init():
    assert lax.eq(
        weights,
        jnp.ones((3, 3, 3, 16), dtype=in_type)
    ).all()

def test_fwd():
    activations = conv.fwd(
        inputs1,
        weights,
        (1, 1),
        preferred_element_type=out_type,
        channel_first=True
    )
    assert lax.eq(
        activations,
        jnp.full((2, 16, 30, 30), 27, dtype=out_type)
    ).all()

    activations = conv.fwd(
        inputs2,
        weights,
        (1, 1),
        preferred_element_type=out_type
    )
    assert lax.eq(
        activations,
        jnp.full((2, 30, 30, 16), 27, dtype=out_type)
    ).all()