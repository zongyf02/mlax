import jax
from jax import (
    numpy as jnp,
    random,
    nn,
    lax
)
import pytest
from mlax.nn import ZNorm
from mlax._test_utils import (
    layer_test_results,
    assert_equal_array,
    assert_close_array
)

def assertClose(x, y, tolerance=1e-4):
    assert (lax.abs(lax.sub(x, y)) < tolerance).all(), f"{x}, {y}"

@pytest.mark.parametrize(
    "config,x,initial_moving_mean,initial_moving_var,expected_train_output,\
        expected_infer_output,expected_moving_mean,expected_moving_var",
    [
        (
            {
                "rng": random.PRNGKey(0),
                "axis": "channel_last",
                "dtype": jnp.float16
            },
            random.normal(random.PRNGKey(7), (4, 16, 16, 3)),
            jnp.zeros((3,), dtype=jnp.float16),
            jnp.ones((3,), dtype=jnp.float16),
            nn.standardize(
                random.normal(random.PRNGKey(7), (4, 16, 16, 3)), (0, 1, 2),
            ),
            random.normal(random.PRNGKey(7), (4, 16, 16, 3)),
            (
                random.normal(random.PRNGKey(7), (4, 16, 16, 3)).mean((0, 1, 2))
                * 0.1
            ).astype(jnp.float16),
            (
                jnp.ones((3,), jnp.float32) * 0.9 +
                random.normal(random.PRNGKey(7), (4, 16, 16, 3)).var((0, 1, 2))
                * 0.1
            ).astype(jnp.float16)
        ),
        (
            {
                "rng": random.PRNGKey(1),                
                "axis": "channel_first",
                "momentum": 0.8,
                "dtype": jnp.float32                
            },
            random.normal(random.PRNGKey(7), (4, 3, 16, 16, 16)),
            jnp.zeros((3,), dtype=jnp.float32),
            jnp.ones((3,), dtype=jnp.float32),
            nn.standardize(
                random.normal(random.PRNGKey(7), (4, 3, 16, 16, 16)),
                (0, 2, 3, 4)
            ),
            random.normal(random.PRNGKey(7), (4, 3, 16, 16, 16)),
            (
                random.normal(random.PRNGKey(7), (4, 3, 16, 16, 16))
                    .mean((0, 2, 3, 4)) * 0.2
            ),
            (
                jnp.ones((3,), jnp.float32) * 0.8 +
                random.normal(random.PRNGKey(7), (4, 3, 16, 16, 16))
                    .var((0, 2, 3, 4)) * 0.2
            )
        ),
        (
            {
                "rng": random.PRNGKey(2),                
                "axis": (0, 1, 3),
                "mean_initializer": nn.initializers.constant(0, dtype=jnp.float16),
                "variance_initializer": nn.initializers.constant(1, dtype=jnp.float16),
                "dtype": jnp.bfloat16
            },
            random.normal(random.PRNGKey(7), (4, 2, 4, 6, 8)),
            jnp.zeros((6,), dtype=jnp.bfloat16),
            jnp.ones((6,), dtype=jnp.bfloat16),
            nn.standardize(
                random.normal(random.PRNGKey(7), (4, 2, 4, 6, 8)), (0, 1, 2, 4)
            ),
            random.normal(random.PRNGKey(7), (4, 2, 4, 6, 8)),
            (
                random.normal(random.PRNGKey(7), (4, 2, 4, 6, 8))
                    .mean((0, 1, 2, 4)) * 0.1
            ).astype(jnp.bfloat16),
            (
                jnp.ones((6,), jnp.float32) * 0.9 +
                random.normal(random.PRNGKey(7), (4, 2, 4, 6, 8))
                    .var((0, 1, 2, 4)) * 0.1
            ).astype(jnp.bfloat16)
        ),
    ]
)
def test_batch_norm(
    config,
    x,
    initial_moving_mean,
    initial_moving_var,
    expected_train_output,
    expected_infer_output,
    expected_moving_mean,
    expected_moving_var
):
    layer, (t_acts, new_t_layer), (i_acts, new_i_layer) = layer_test_results(
        ZNorm, config, x
    )
    assert_equal_array(layer.moving_mean.data, initial_moving_mean)
    assert_equal_array(layer.moving_var.data, initial_moving_var)

    assert_close_array(t_acts, expected_train_output)
    assert_close_array(new_t_layer.moving_mean.data, expected_moving_mean)
    assert_close_array(new_t_layer.moving_var.data, expected_moving_var)

    assert_close_array(i_acts, expected_infer_output)
    assert_equal_array(new_i_layer.moving_mean.data, initial_moving_mean)
    assert_equal_array(new_i_layer.moving_var.data, initial_moving_var)
