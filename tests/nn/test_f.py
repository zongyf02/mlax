import pytest
from jax import (
    numpy as jnp,
    random,
    lax
)
from mlax.nn import F
from mlax._test_utils import (
    layer_test_results,
    assert_equal_array,
    assert_equal_pytree
)

@pytest.mark.parametrize(
    "config,x,expected_train_output,expected_infer_output",
    [
        (
            {
                "train_fn": lambda x: x,
                "infer_fn": lambda x: 2 * x
            },
            jnp.ones((2, 4, 4, 3), jnp.bfloat16),
            jnp.ones((2, 4, 4, 3), jnp.bfloat16),
            jnp.full((2, 4, 4, 3), 2, jnp.bfloat16)
        ),
        (
            {
                "train_fn": lambda x: x
            },
            jnp.ones((2, 4, 4, 3), jnp.float16),
            jnp.ones((2, 4, 4, 3), jnp.float16),
            jnp.ones((2, 4, 4, 3), jnp.float16)
        ),
        (
            {
                "train_fn": lax.psum,
                "infer_fn": lax.pmax
            },
            jnp.ones((2, 4), jnp.float32),
            jnp.full((2, 4), 2, jnp.float32),
            jnp.ones((2, 4), jnp.float32)
        ),
    ]
)
def test_f(config, x, expected_train_output, expected_infer_output):
    _, (t_acts, _), (i_acts, _) = layer_test_results(
        F, config, x
    )
    assert_equal_array(t_acts, expected_train_output)
    assert_equal_array(i_acts, expected_infer_output)

@pytest.mark.parametrize(
    "config,x,rng,expected_train_output,expected_infer_output",
    [
        (
            {
                "train_fn": lambda x, rng: (x, rng),
                "infer_fn": lambda x, rng: (2 * x, rng)
            },
            jnp.ones((2, 4, 4, 3), jnp.bfloat16),
            random.PRNGKey(0),
            (jnp.ones((2, 4, 4, 3), jnp.bfloat16), random.PRNGKey(0)),
            (jnp.full((2, 4, 4, 3), 2, jnp.bfloat16), random.PRNGKey(0))
        ),
        (
            {
                "train_fn": lambda x, rng: (x, rng)
            },
            jnp.ones((2, 4, 4, 3), jnp.float16),
            random.PRNGKey(1),
            (jnp.ones((2, 4, 4, 3), jnp.float16), random.PRNGKey(1)),
            (jnp.ones((2, 4, 4, 3), jnp.float16), random.PRNGKey(1))
        ),
        (
            {
                "train_fn": lambda x, rng, axis_name: (lax.psum(x, axis_name), rng),
                "infer_fn": lambda x, rng, axis_name: (lax.pmax(x, axis_name), rng)
            },
            jnp.ones((2, 4), jnp.float32),
            random.PRNGKey(2),
            (jnp.full((2, 4), 2, jnp.float32), random.PRNGKey(2)),
            (jnp.ones((2, 4), jnp.float32), random.PRNGKey(2))
        ),
    ]
)
def test_f_rng(config, x, rng, expected_train_output, expected_infer_output):
    _, (t_acts, _), (i_acts, _) = layer_test_results(
        F, config, x, rng=rng, y_vmap_axis=(0, None)
    )
    assert_equal_pytree(t_acts, expected_train_output)
    assert_equal_pytree(i_acts, expected_infer_output)
