from mlax import optim
import jax

def test_apply_updates():
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    update_gradients = (
        jax.random.normal(key1, (4, 3), dtype="bfloat16"),
        jax.random.normal(key2, (4,), dtype="bfloat16")
    )
    model_weights = (
        jax.numpy.zeros((4, 3), dtype="bfloat16"),
        jax.numpy.zeros((4,), dtype="bfloat16")
    )
    new_model_weights = optim.apply_updates(
        update_gradients,
        model_weights,
        minimize=False
    )
    assert (
        jax.tree_util.tree_structure(model_weights) ==
        jax.tree_util.tree_structure(new_model_weights)
    )
    assert jax.lax.eq(
        new_model_weights[0],
        new_model_weights[0]
    ).all()
    assert jax.lax.eq(
        new_model_weights[1],
        new_model_weights[1]
    ).all()
