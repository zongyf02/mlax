import pytest
from jax import (
    random,
    numpy as jnp
)
from mlax.nn import Recurrent, F
from mlax._test_utils import (
    layer_test_results,
    assert_equal_pytree
)

@pytest.mark.parametrize(
    "config,xh,expected_train_output,expected_infer_output",
    [
        (
            {
                "cell": F(
                    lambda xh: (xh[0] + xh[1], xh[1] + 1),
                    lambda xh: xh
                ),
                "reverse": False,
                "unroll": 1
            },
            (
                jnp.expand_dims(jnp.arange(10, dtype=jnp.float16), 0),
                jnp.ones((1,), dtype=jnp.float32)
            ),
            (
                jnp.expand_dims(jnp.arange(1, 21, 2, dtype=jnp.float32), 0),
                jnp.full((1,), 11, dtype=jnp.float32)
            ),
            (
                jnp.expand_dims(jnp.arange(10, dtype=jnp.float16), 0),
                jnp.ones((1,), dtype=jnp.float32)
            )
        ),
        (
            {
                "cell": F(
                    lambda xh: xh,
                    lambda xh: ((xh[0] + xh[1], xh[0] + xh[1] + 1), xh[1] + 2)
                ),
                "reverse": True,
                "unroll": 2
            },
            (
                jnp.expand_dims(jnp.arange(10, dtype=jnp.int16), 0),
                jnp.ones((1,), dtype=jnp.int8)
            ),
            (
                jnp.expand_dims(jnp.arange(10, dtype=jnp.int16), 0),
                jnp.ones((1,), dtype=jnp.int8)
            ),
            (
                (
                    jnp.expand_dims(jnp.arange(19, 9, -1, dtype=jnp.int16), 0),
                    jnp.expand_dims(jnp.arange(20, 10, -1, dtype=jnp.int16), 0),
                ),
                jnp.full((1,), 21, dtype=jnp.int8)
            )
        ),
    ]
)
def test_recurrent(
    config, xh, expected_train_output, expected_infer_output
):
    layer, (t_acts, _), (i_acts, _) = layer_test_results(
        Recurrent, config, xh
    )
    assert layer.cell.is_set_up is True

    assert_equal_pytree(t_acts, expected_train_output)
    assert_equal_pytree(i_acts, expected_infer_output)

@pytest.mark.parametrize(
    "config,xh,rng,expected_train_output,expected_infer_output",
    [
        (
            {
                "cell": F(
                    lambda xh, rng: ((xh[0] + xh[1], rng), xh[1] + 2),
                    lambda xh, rng: ((xh[0], rng), xh[1])
                ),
                "reverse": False,
                "unroll": 1
            },
            (
                jnp.expand_dims(jnp.arange(10, dtype=jnp.float32), 0),
                jnp.ones((1,), dtype=jnp.bfloat16)
            ),
            random.PRNGKey(7),
            (
                (
                    jnp.expand_dims(jnp.arange(1, 29, 3, dtype=jnp.float32), 0),
                    jnp.stack([
                        random.fold_in(random.PRNGKey(7), i) for i in range(10)
                    ], 0)
                ),
                jnp.full((1,), 21, dtype=jnp.bfloat16)
            ),
            (
                (
                    jnp.expand_dims(jnp.arange(10, dtype=jnp.float32), 0),
                    jnp.stack([
                        random.fold_in(random.PRNGKey(7), i) for i in range(10)
                    ], 0)
                ),
                jnp.ones((1,), dtype=jnp.bfloat16)
            )
        ),
        (
            {
                "cell": F(
                    lambda xh, rng: ((xh[0], rng), xh[1]),
                    lambda xh, rng: ((xh[0] + xh[1], rng), xh[1] + 2)
                ),
                "reverse": True,
                "unroll": 2
            },
            (
                jnp.expand_dims(jnp.arange(10, dtype=jnp.int8), 0),
                jnp.ones((1,), dtype=jnp.int16)
            ),
            random.PRNGKey(7),
            (
                (
                    jnp.expand_dims(jnp.arange(10, dtype=jnp.int8), 0),
                    jnp.stack([
                        random.fold_in(random.PRNGKey(7), i)
                        for i in range(9, -1, -1)
                    ], 0)
                ),
                jnp.ones((1,), dtype=jnp.int16)
            ),
            (
                (
                    jnp.expand_dims(jnp.arange(19, 9, -1, dtype=jnp.int16), 0),
                    jnp.stack([
                        random.fold_in(random.PRNGKey(7), i)
                        for i in range(9, -1, -1)
                    ], 0)
                ),
                jnp.full((1,), 21, dtype=jnp.int16)
            )
        )
    ]
)
def test_recurrent_rng(
    config, xh, rng, expected_train_output, expected_infer_output
):
    layer, (t_acts, _), (i_acts, _) = layer_test_results(
        Recurrent, config, xh, rng, y_vmap_axis=((0, None), 0)
    )
    assert layer.cell.is_set_up is True

    assert_equal_pytree(t_acts, expected_train_output)
    assert_equal_pytree(i_acts, expected_infer_output)
