import pytest
from jax import (
    numpy as jnp,
    random,
    nn
)
from mlax.nn import Series, SeriesRng, Linear, Bias, F, FRng
from mlax._test_utils import (
    layer_test_results,
    assert_equal_array,
    assert_equal_pytree
)

@pytest.mark.parametrize(
    "layers,x,expected_train_output,expected_infer_output",
    [
        (
            (
                Linear(
                    random.PRNGKey(0),
                    out_features=3,
                    kernel_initializer=nn.initializers.ones
                ),
                Bias(
                    random.PRNGKey(1),
                    in_features=-1,
                    bias_initializer=nn.initializers.ones
                ),
                F(
                    train_fn=lambda x: x,
                    infer_fn=lambda x: 2 * x
                )
            ),
            jnp.ones((2, 4), jnp.bfloat16),
            jnp.full((2, 3), 5, jnp.bfloat16),
            jnp.full((2, 3), 10, jnp.bfloat16),
        )
    ]
)
def test_series(
    layers, x, expected_train_output, expected_infer_output
):
    model, (t_acts, new_t_model), (i_acts, new_i_model) = layer_test_results(
        Series, {"layers": layers}, x
    )
    assert model.layers.trainable is None
    assert isinstance(model.layers.data, list)

    assert_equal_array(t_acts, expected_train_output)
    assert new_t_model.layers.trainable is None
    assert isinstance(new_t_model.layers.data, list)

    assert_equal_array(i_acts, expected_infer_output)
    assert new_i_model.layers.trainable is None
    assert isinstance(new_i_model.layers.data, list)

@pytest.mark.parametrize(
    "layers,x,rng,expected_train_output,expected_infer_output",
    [
        (
            iter([
                Linear(
                    random.PRNGKey(0),
                    out_features=3,
                    kernel_initializer=nn.initializers.ones
                ),
                Bias(
                    random.PRNGKey(1),
                    in_features=-1,
                    bias_initializer=nn.initializers.ones
                ),
                FRng(
                    train_fn=lambda x, rng: (x, rng),
                    infer_fn=lambda x, rng: (2 * x, rng)
                )
            ]),
            jnp.ones((2, 4), jnp.bfloat16),
            random.PRNGKey(7),
            (jnp.full((2, 3), 5, jnp.bfloat16), random.PRNGKey(7)),
            (jnp.full((2, 3), 10, jnp.bfloat16), random.PRNGKey(7))
        ),
    ]
)
def test_series_rng(
    layers, x, rng, expected_train_output, expected_infer_output
):
    model, (t_acts, new_t_model), (i_acts, new_i_model) = layer_test_results(
        SeriesRng, {"layers": layers}, x, rng=rng, y_vmap_axis=(0, None)
    )
    assert model.layers.trainable is None
    assert isinstance(model.layers.data, list)

    assert_equal_pytree(t_acts, expected_train_output)
    assert new_t_model.layers.trainable is None
    assert isinstance(new_t_model.layers.data, list)

    assert_equal_pytree(i_acts, expected_infer_output)
    assert new_i_model.layers.trainable is None
    assert isinstance(new_i_model.layers.data, list)
