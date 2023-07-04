import pytest
from jax import (
    numpy as jnp,
    random,
    nn
)
from mlax.nn import Parallel, ParallelRng, Scaler, F, FRng
from mlax._test_utils import (
    layer_test_results,
    assert_equal_pytree
)

@pytest.mark.parametrize(
    "layers,x,expected_train_output,expected_infer_output",
    [
        (
            [
                Scaler(
                    random.PRNGKey(0),
                    in_features=-1,
                    scaler_initializer=nn.initializers.constant(2)
                ),
                F(train_fn=lambda x: x, infer_fn=lambda x: 2 * x),
                F(train_fn=lambda x: 3 * x)
            ],
            [
                jnp.ones((2, 4), jnp.bfloat16),
                jnp.ones((2, 3), jnp.bfloat16),
                jnp.ones((2, 2), jnp.float32)
            ],
            (
                jnp.full((2, 4), 2, jnp.bfloat16),
                jnp.ones((2, 3), jnp.bfloat16),
                jnp.full((2, 2), 3, jnp.float32)
            ),
            (
                jnp.full((2, 4), 2, jnp.bfloat16),
                jnp.full((2, 3), 2, jnp.bfloat16),
                jnp.full((2, 2), 3, jnp.float32)
            )
        ),
    ]
)
def test_parallel(
    layers, x, expected_train_output, expected_infer_output
):
    model, (t_acts, new_t_model), (i_acts, new_i_model) = layer_test_results(
        Parallel, {"layers": layers}, x
    )
    assert model.layers.trainable is None
    assert isinstance(model.layers.data, list)

    assert_equal_pytree(t_acts, expected_train_output)
    assert new_t_model.layers.trainable is None
    assert isinstance(new_t_model.layers.data, list)

    assert_equal_pytree(i_acts, expected_infer_output)
    assert new_i_model.layers.trainable is None
    assert isinstance(new_i_model.layers.data, list)

@pytest.mark.parametrize(
    "layers,x,rng,expected_train_output,expected_infer_output",
    [
        (
            iter((
                Scaler(
                    random.PRNGKey(0),
                    in_features=-1,
                    scaler_initializer=nn.initializers.constant(2)
                ),
                F(train_fn=lambda x: x, infer_fn=lambda x: 2 * x
                ),
                FRng(train_fn=lambda x, rng: (3 * x, rng))
            )),
            (
                jnp.ones((2, 4), jnp.bfloat16),
                jnp.ones((2, 3), jnp.bfloat16),
                jnp.ones((2, 2), jnp.float32)
            ),
            random.PRNGKey(7),
            (
                jnp.full((2, 4), 2, jnp.bfloat16),
                jnp.ones((2, 3), jnp.bfloat16),
                (jnp.full((2, 2), 3, jnp.float32), random.PRNGKey(7))
            ),
            (
                jnp.full((2, 4), 2, jnp.bfloat16),
                jnp.full((2, 3), 2, jnp.bfloat16),
                (jnp.full((2, 2), 3, jnp.float32), random.PRNGKey(7))
            )
        ),
    ]
)
def test_parallel_rng(
    layers, x, rng, expected_train_output, expected_infer_output
):
    model, (t_acts, new_t_model), (i_acts, new_i_model) = layer_test_results(
        ParallelRng, {"layers": layers}, x, rng=rng,
        y_vmap_axis=(0, 0, (0, None))
    )
    assert model.layers.trainable is None
    assert isinstance(model.layers.data, list)

    assert_equal_pytree(t_acts, expected_train_output)
    assert new_t_model.layers.trainable is None
    assert isinstance(new_t_model.layers.data, list)

    assert_equal_pytree(i_acts, expected_infer_output)
    assert new_i_model.layers.trainable is None
    assert isinstance(new_i_model.layers.data, list)
