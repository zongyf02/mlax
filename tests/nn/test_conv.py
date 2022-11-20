from mlax.nn import conv_f, conv_l
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn
)

key1, key2 = random.split(random.PRNGKey(0))
in_type = jnp.float16
out_type = jnp.float32
inputs1 = jnp.ones((4, 3, 32, 32), dtype=in_type)
inputs2 = jnp.ones((4, 32, 32, 3), dtype=in_type)
weights1 = conv_f.init(
    key1,
    in_channels=3,
    out_channels=16,
    filter_shape=(5, 5),
    kernel_initializer=nn.initializers.constant(1, dtype=out_type),
    dtype=in_type
)
weights2 = conv_l.init(
    key2,
    in_channels=3,
    out_channels=16,
    filter_shape=(5, 5),
    kernel_initializer=nn.initializers.constant(1, dtype=out_type),
    dtype=in_type
)

def test_init():
    assert lax.eq(
        weights1,
        jnp.ones((16, 3, 5, 5), dtype=in_type)
    ).all()
    assert lax.eq(
        weights2,
        jnp.ones((16, 5, 5, 3), dtype=in_type)
    ).all()

def test_fwd():
    activations = conv_f.fwd(
        inputs1,
        weights1,
        (1, 1),
        preferred_element_type=out_type
    )
    assert lax.eq(
        activations,
        jnp.full((4, 16, 28, 28), 75, dtype=out_type)
    ).all()

    activations = conv_l.fwd(
        inputs2,
        weights2,
        (1, 1),
        preferred_element_type=out_type
    )
    assert lax.eq(
        activations,
        jnp.full((4, 28, 28, 16), 75, dtype=out_type)
    ).all()