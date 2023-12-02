import pytest
import jax
from jax import (
    random,
    nn,
    numpy as jnp
)
from mlax.nn.functional import rms_norm
from mlax._test_utils import assert_close_array

def ref_rms_norm(x, axis):
    rms = jnp.sqrt(jnp.mean(x**2, axis, keepdims=True))
    return x / rms

@pytest.mark.parametrize(
    "input,axis,batch_axis_name,expected_output",
    [
        (
            random.normal(random.PRNGKey(0), (1, 4, 8)),
            (0, 1),
            (),
            ref_rms_norm(random.normal(random.PRNGKey(0), (1, 4, 8)), (1, 2))
        ),
        (
            random.normal(random.PRNGKey(1), (1, 3, 4, 4, 4)),
            [0, 1, 2, 3],
            (),
            ref_rms_norm(
                random.normal(random.PRNGKey(1), (1, 3, 4, 4, 4)), (1, 2, 3, 4)
            )
        ),
        (
            random.normal(random.PRNGKey(2), (4, 1, 8, 4)),
            "channel_first",
            (),
            ref_rms_norm(random.normal(random.PRNGKey(2), (4, 1, 8, 4)), (2, 3))
        ),
        (
            random.normal(random.PRNGKey(3), (1, 4, 4, 3)),
            "channel_last",
            (),
            ref_rms_norm(random.normal(random.PRNGKey(3), (1, 4, 4, 3)), (1, 2))
        ),
        (
            random.normal(random.PRNGKey(4), (1, 3, 4, 4)),
            2,
            "N",
            ref_rms_norm(random.normal(random.PRNGKey(4), (1, 3, 4, 4)), (0, 3))
        )
    ]
)
def test_rms_norm(input, axis, batch_axis_name, expected_output):
    activations = jax.vmap(
        rms_norm, in_axes=(0, None, None), axis_name=batch_axis_name
    )(input, axis, batch_axis_name)
    assert_close_array(activations, expected_output)
