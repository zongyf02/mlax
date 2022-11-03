from mlax.nn import bias
import jax

inputs = jax.numpy.zeros((4, 3), dtype="bfloat16")

weights = bias.init(
    jax.random.PRNGKey(0),
    (4, 3),
    bias_initializer=jax.nn.initializers.constant(1, dtype="float64"),
    dtype = "bfloat16" # Should override kernel initializer's dtype
)

def test_init():
    assert jax.lax.eq(
        weights,
        jax.numpy.ones((4, 3), dtype="bfloat16")
    ).all()

def test_fwd():
    activations = bias.fwd(
        inputs, weights
    )
    assert jax.lax.eq(
        weights,
        jax.numpy.ones((4,3), dtype="bfloat16")
    ).all()
