from mlax.nn import linear
import jax

inputs = jax.numpy.ones((4,), dtype="bfloat16")

weights = linear.init(
    jax.random.PRNGKey(0),
    in_features=4, out_features=3,
    kernel_initializer=jax.nn.initializers.constant(1, dtype="float64"),
    dtype="bfloat16" # Should override kernel initializer's dtype
)

def test_init():
    assert jax.lax.eq(
        weights,
        jax.numpy.ones((4, 3), dtype="bfloat16")
    ).all()

def test_fwd():
    activations = linear.fwd(
        inputs, weights,
        preferred_element_type="float32" # accumulation type
    )
    assert jax.lax.eq(
        activations,
        jax.numpy.full((3,), 4, dtype="float32")
    ).all()
