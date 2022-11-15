from mlax.nn import conv
import jax.numpy as jnp
from jax import (
    random,
    lax,
    nn
)

input_dtype = "float32"
inputs = jnp.ones((2, 3, 32, 32), dtype=input_dtype)
weights = conv.init(
    random.PRNGKey(0),
    in_channels=3,
    out_channels=16,
    filter_shape=(3, 3),
    kernel_spec="HWIO",
    kernel_initializer=nn.initializers.ones,
    dtype=input_dtype
)

def test_init():
    assert lax.eq(
        weights,
        jnp.ones((3, 3, 3, 16), dtype=input_dtype)
    ).all()

def test_fwd():
    activations = conv.fwd(
        inputs,
        weights,
        (1, 1),
        dimension_numbers=("NCHW", "HWIO", "NHWC"),
        preferred_element_type="float32"
    )
    assert lax.eq(
        activations,
        jnp.full((2, 30, 30, 16), 27, dtype="float32")
    ).all()
