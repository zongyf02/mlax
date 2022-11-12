from mlax import optim
import jax.numpy as jnp
from jax import (
    random,
    lax,
    tree_util
)

def test_apply_updates():
    key1, key2 = random.split(random.PRNGKey(0))
    update_gradients = (
        random.normal(key1, (4, 3), dtype="bfloat16"),
        random.normal(key2, (4,), dtype="bfloat16")
    )
    model_weights = (
        jnp.zeros((4, 3), dtype="bfloat16"),
        jnp.zeros((4,), dtype="bfloat16")
    )
    new_model_weights = optim.apply_updates(
        update_gradients,
        model_weights,
        minimize=False
    )
    assert (
        tree_util.tree_structure(model_weights) ==
        tree_util.tree_structure(new_model_weights)
    )
    assert lax.eq(
        new_model_weights[0],
        new_model_weights[0]
    ).all()
    assert lax.eq(
        new_model_weights[1],
        new_model_weights[1]
    ).all()
