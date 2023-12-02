"""Testing utilities."""
import jax
from jax import (
    numpy as jnp,
    lax,
    tree_util as jtu
)

def assert_close_array(a, b, threshold=1e-04):
    """Assert two arrays have similar values."""
    assert lax.bitwise_or(
        lax.bitwise_and(jnp.isnan(a), jnp.isnan(b)), # NaN are not equal to NaN
        (lax.abs(lax.sub(a, b)) < threshold)
    ).all()

def assert_equal_array(a, b):
    """Assert two arrays have the same values."""
    assert lax.bitwise_or(
        lax.bitwise_and(jnp.isnan(a), jnp.isnan(b)), # NaN are not equal to NaN
        lax.eq(a, b)
    ).all()

def assert_equal_pytree(a, b):
    """Assert two PyTrees contain equal arrays."""
    jtu.tree_map(assert_equal_array, a, b)

def layer_test_results(
    cls, config, x, rng=None, batch_axis_name="N", x_vmap_axis=0, y_vmap_axis=0
):
    """Return an initilized test layer and results of a training and inference
    calls.
    """
    layer = cls(**config)
    assert layer.is_set_up is False

    fwd_jit = jax.jit(
        jax.vmap(
            cls.__call__,
            in_axes = (None, x_vmap_axis, None, None, None),
            out_axes = (y_vmap_axis, None),
            axis_name=batch_axis_name
        ),
        static_argnames=("self", "inference_mode", "batch_axis_name")
    )

    train_init_acts, init_layer = fwd_jit(
        layer,
        x,
        rng,
        False, # inference_mode
        batch_axis_name # batch axis name
    )
    assert init_layer.is_set_up is True

    infer_init_acts, init_layer = fwd_jit(
        layer,
        x,
        rng,
        True, # inference_mode
        batch_axis_name # batch axis name
    )
    assert init_layer.is_set_up is True

    train_acts, new_train_layer = fwd_jit(
        init_layer,
        x,
        rng,
        False, # inference_mode
        batch_axis_name # batch axis name
    )
    assert_equal_pytree(train_acts, train_init_acts)

    infer_acts, new_infer_layer = fwd_jit(
        init_layer,
        x,
        rng,
        True, # inference_mode
        batch_axis_name # batch axis name
    )
    assert_equal_pytree(infer_acts, infer_init_acts)

    return (
        init_layer, (train_acts, new_train_layer), (infer_acts, new_infer_layer)
    )
