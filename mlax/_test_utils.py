import jax
from jax import (
    numpy as jnp,
    lax,
    tree_util as jtu
)

def assert_close_array(a, b, threshold=1e-04):
    assert (lax.abs(lax.sub(a, b)) < threshold).all()

def assert_equal_array(a, b):
    assert lax.bitwise_or(
        lax.bitwise_and(jnp.isnan(a), jnp.isnan(b)), # NaN are not equal to NaN
        lax.eq(a, b)
    ).all()

def assert_equal_pytree(a, b):
    jtu.tree_map(assert_equal_array, a, b)

def layer_test_results(
    cls, config, x, rng=None, batch_axis_name="N", x_vmap_axis=0, y_vmap_axis=0
):
    layer = cls(**config)
    assert layer.initialized is False
    _, layer = layer(
        jtu.tree_map(lambda x: x[0], x), rng, inference_mode=True
    )
    assert layer.initialized is True

    fwd_jit = jax.jit(
        jax.vmap(
            cls.__call__,
            in_axes = (None, x_vmap_axis, None, None, None),
            out_axes = (y_vmap_axis, None),
            axis_name=batch_axis_name
        ),
        static_argnames=("self", "inference_mode", "batch_axis_name")
    )

    train_acts, new_train_layer = fwd_jit(
        layer,
        x,
        rng,
        False, # inference_mode
        batch_axis_name # batch axis name
    )

    infer_acts, new_infer_layer = fwd_jit(
        layer,
        x,
        rng,
        True, # inference_mode
        batch_axis_name # batch axis name
    )

    return layer, (train_acts, new_train_layer), (infer_acts, new_infer_layer)
