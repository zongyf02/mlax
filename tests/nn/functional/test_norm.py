from mlax.nn.functional import (
    layer_norm, instance_norm, group_norm
)
import jax.numpy as jnp
from jax import (
    lax,
    random,
    nn
)
import pytest

@pytest.mark.parametrize(
    "input",
    [
        random.normal(random.PRNGKey(0), (4, 8), jnp.float32),
        random.normal(random.PRNGKey(1), (3, 4, 4, 4), jnp.float32),
    ]
)
def test_layer_norm(input):
    activations = layer_norm(input)
    assert (
        lax.abs(
            lax.sub(
                activations,
                nn.standardize(input, range(input.ndim))
            )
        ) < 1e-05
    ).all()

@pytest.mark.parametrize(
    "input,channel_last",
    [
        (
            random.normal(random.PRNGKey(0), (4, 8), jnp.float32),
            False,
        ),
        (
            random.normal(random.PRNGKey(1), (4, 4, 4, 3), jnp.float32),
            True
        )
    ]
)
def test_instance_norm(input, channel_last):
    activations = instance_norm(input, channel_last)
    dims = range(input.ndim - 1) if channel_last else range(1, input.ndim)
    assert (
        lax.abs(
            lax.sub(
                activations,
                nn.standardize(input, dims)
            )
        ) < 1e-05
    ).all()

@pytest.mark.parametrize(
    "input,num_groups,channel_last",
    [
        (
            random.normal(random.PRNGKey(0), (4, 8), jnp.float32),
            1,
            False,
        ),
        (
            random.normal(random.PRNGKey(1), (4, 4, 4, 3), jnp.float32),
            1,
            True
        ),
        (
            random.normal(random.PRNGKey(2), (4, 8), jnp.float32),
            4,
            False,
        ),
        (
            random.normal(random.PRNGKey(3), (4, 4, 4, 3), jnp.float32),
            3,
            True
        ),
        (
            random.normal(random.PRNGKey(4), (4, 8), jnp.float32),
            2,
            False,
        ),
        (
            random.normal(random.PRNGKey(5), (4, 8), jnp.float32),
            2,
            True,
        ),
    ]
)
def test_group_norm(input, num_groups, channel_last):
    def _jnp_group_norm(input, num_groups, channel_last):
        if num_groups == 1:
            return nn.standardize(input, range(input.ndim)) # Layer norm

        # Instance norm
        if channel_last and input.shape[-1] == num_groups:
            return nn.standardize(input, range(input.ndim - 1))
        if not channel_last and input.shape[0] == num_groups:
            return nn.standardize(input, range(1, input.ndim))

        in_shape = input.shape
        if channel_last:
            input = jnp.reshape(input, (*in_shape[:-1], -1, num_groups))
            dims = range(input.ndim - 1)
        else:
            input = jnp.reshape(input, (num_groups, -1, *in_shape[1:]))
            dims = range(1, input.ndim)
        
        input = nn.standardize(input, dims)
        return jnp.reshape(input, in_shape)

    activations = group_norm(input, num_groups, channel_last)
    assert (
        lax.abs(
            lax.sub(
                activations,
                _jnp_group_norm(input, num_groups, channel_last)
            )
        ) < 1e-05
    ).all()