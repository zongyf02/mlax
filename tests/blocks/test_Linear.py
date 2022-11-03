from mlax.blocks import Linear
import jax

inputs = jax.numpy.ones((4,), dtype="bfloat16")

weights = Linear.init(
    jax.random.PRNGKey(0),
    in_features=4, out_features=3,
    kernel_initializer=jax.nn.initializers.constant(1, dtype="float64"),
    bias_initializer=jax.nn.initializers.ones,
    dtype="bfloat16" # Should override kernel initializer's dtype
)

def test_init():
    assert jax.lax.eq(
        weights.kernel,
        jax.numpy.ones((4, 3), dtype="bfloat16")
    ).all()
    assert jax.lax.eq(
        weights.bias,
        jax.numpy.ones((3,), dtype="bfloat16")
    ).all()

def test_fwd():
    activations = Linear.fwd(
        inputs, weights 
    )
    assert jax.lax.eq(
        activations,
        jax.numpy.full((3,), 5, dtype="bfloat16")
    ).all()
