import pytest
import jax
from jax import (
    random,
    nn
)
from mlax.nn.functional import z_norm
from mlax._test_utils import assert_close_array

@pytest.mark.parametrize(
    "input,axis,batch_axis_name,expected_output",
    [
        (
            random.normal(random.PRNGKey(0), (1, 4, 8)),
            "all",
            (),
            nn.standardize(random.normal(random.PRNGKey(0), (1, 4, 8)), (1, 2))
        ),
        (
            random.normal(random.PRNGKey(1), (1, 3, 4, 4, 4)),
            [0, 1, 2, 3],
            (),
            nn.standardize(
                random.normal(random.PRNGKey(1), (1, 3, 4, 4, 4)), (1, 2, 3, 4)
            )
        ),
        (
            random.normal(random.PRNGKey(2), (4, 1, 8, 4)),
            "channel_first",
            (),
            nn.standardize(random.normal(random.PRNGKey(2), (4, 1, 8, 4)), (2, 3))
        ),
        (
            random.normal(random.PRNGKey(3), (1, 4, 4, 3)),
            "channel_last",
            (),
            nn.standardize(random.normal(random.PRNGKey(3), (1, 4, 4, 3)), (1, 2))
        ),
        (
            random.normal(random.PRNGKey(4), (1, 4, 3, 4)),
            (0, 2),
            "N",
            nn.standardize(random.normal(random.PRNGKey(4), (1, 4, 3, 4)), (0, 1, 3))
        ),
        (
            random.normal(random.PRNGKey(5), (1, 3, 4, 4)),
            2,
            "N",
            nn.standardize(random.normal(random.PRNGKey(5), (1, 3, 4, 4)), (0, 3))
        )
    ]
)
def test_z_norm(input, axis, batch_axis_name, expected_output):
    activations = jax.vmap(
        z_norm, in_axes=(0, None, None), axis_name=batch_axis_name
    )(input, axis, batch_axis_name)
    assert_close_array(activations, expected_output)
