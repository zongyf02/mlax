import jax
from jax import (
    numpy as jnp,
    random,
    nn,
    lax
)
from mlax import (
    fwd,
    is_trainable
)
from mlax.nn import BatchNorm
import pytest

@pytest.mark.parametrize(
    "config,batch_axis_name,input,inference_mode,expected_output,expected_moving_mean,expected_moving_var",
    [
        (
            {
                "rng": random.PRNGKey(0),
                "batch_axis_name": "batch",
                "channel_last": True,
                "dtype": jnp.bfloat16
            },
            "batch",
            random.normal(random.PRNGKey(0), (4, 16, 16, 3)),
            False,
            nn.standardize(
                random.normal(random.PRNGKey(0), (4, 16, 16, 3)), (0, 1, 2),
            ),
            (
                random.normal(random.PRNGKey(0), (4, 16, 16, 3)).mean((0, 1, 2))
                * 0.1
            ).astype(jnp.bfloat16),
            (
                jnp.ones((3,), jnp.float32) * 0.9 +
                random.normal(random.PRNGKey(0), (4, 16, 16, 3)).var((0, 1, 2))
                * 0.1
            ).astype(jnp.bfloat16)
        ),
        (
            {
                "rng": random.PRNGKey(1),
                "batch_axis_name": "batch",
                "momentum": 0.8,
                "channel_last": False
            },
            "batch",
            random.normal(random.PRNGKey(1), (4, 3, 16, 16, 16)),
            False,
            nn.standardize(
                random.normal(random.PRNGKey(1), (4, 3, 16, 16, 16)),
                (0, 2, 3, 4)
            ),
            (
                random.normal(random.PRNGKey(1), (4, 3, 16, 16, 16)).mean(
                    (0, 2, 3, 4)
                ) * 0.2
            ),
            (
                jnp.ones((3,), jnp.float32) * 0.8 +
                random.normal(random.PRNGKey(1), (4, 3, 16, 16, 16)).var(
                    (0, 2, 3, 4)
                ) * 0.2
            )
        ),
        (
            {
                "rng": random.PRNGKey(2),
                "batch_axis_name": "batch0",
                "channel_last": False,
                "dtype": jnp.bfloat16
            },
            "batch0",
            jnp.ones((4, 3, 16), jnp.float32),
            True,
            jnp.ones((4, 3, 16), jnp.float32),
            jnp.zeros((3,), jnp.bfloat16),
            jnp.ones((3,), jnp.bfloat16),
        ),
    ]
)
def test_batch_norm(
    config,
    batch_axis_name,
    input,
    inference_mode,
    expected_output,
    expected_moving_mean,
    expected_moving_var
):
    batch_norm = BatchNorm(
        **config
    )
    assert batch_norm.moving_mean.data is None
    assert batch_norm.moving_var.data is None

    fwd_jit = jax.jit(
        jax.vmap(
            fwd,
            in_axes = (None, None, 0, None, None),
            out_axes = (0, None),
            axis_name = batch_axis_name
        ),
        static_argnames="inference_mode"
    )

    activations, batch_norm = fwd_jit(
        *batch_norm.partition(),
        input,
        None, # rng
        inference_mode # inference_mode
    )
    assert (
        lax.abs(
            lax.sub(
                activations,
                expected_output
            )
        ) < 1e-5
    ).all()
    assert (
        lax.abs(
            lax.sub(
                batch_norm.moving_mean.data,
                expected_moving_mean
            )
        ) < 1e-4
    ).all()
    assert (
        lax.abs(
            lax.sub(
                batch_norm.moving_var.data,
                expected_moving_var
            )
        ) < 1e-4
    ).all()
