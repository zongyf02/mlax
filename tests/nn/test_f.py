import jax
from jax import (
    numpy as jnp,
    random,
    lax
)
from mlax import (
    fwd,
    is_trainable
)
from mlax.nn import F, FRng
import pytest

@pytest.mark.parametrize(
    "config,input,expected_train_output,expected_infer_output",
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
            jnp.ones((2, 4, 4, 3), jnp.bfloat16),
            jnp.ones((2, 4, 4, 3), jnp.bfloat16),
            jnp.ones((2, 4, 4, 3), jnp.bfloat16)
        ),
    ]
)
def test_f(
    config,
    input,
    expected_train_output,
    expected_infer_output
):
    f = F(
        **config
    )

    fwd_jit = jax.jit(
        jax.vmap(
            fwd,
            in_axes = (None, None, 0, None, None),
            out_axes = (0, None)
        ),
        static_argnames="inference_mode"
    )

    activations, f = fwd_jit(
        *f.partition(is_trainable),
        input,
        None, # rng
        False # inference_mode
    )
    assert lax.eq(
        activations,
        expected_train_output
    ).all()

    activations, f = fwd_jit(
        *f.partition(is_trainable),
        input,
        None, # rng
        True # inference_mode
    )
    assert lax.eq(
        activations,
        expected_infer_output
    ).all()

@pytest.mark.parametrize(
    "config,input,rng,expected_train_output,expected_infer_output",
    [
        (
            {
                "train_fn": lambda x, rng: (x, rng),
                "infer_fn": lambda x, rng: (2 * x, rng)
            },
            jnp.ones((2, 4, 4, 3), jnp.bfloat16),
            random.PRNGKey(1),
            jnp.ones((2, 4, 4, 3), jnp.bfloat16),
            jnp.full((2, 4, 4, 3), 2, jnp.bfloat16)
        ),
        (
            {
                "train_fn": lambda x, rng: (x, rng)
            },
            jnp.ones((2, 4, 4, 3), jnp.bfloat16),
            random.PRNGKey(2),
            jnp.ones((2, 4, 4, 3), jnp.bfloat16),
            jnp.ones((2, 4, 4, 3), jnp.bfloat16)
        ),
    ]
)
def test_f_rng(
    config,
    input,
    rng,
    expected_train_output,
    expected_infer_output
):
    f_rng = FRng(
        **config
    )

    fwd_jit = jax.jit(
        jax.vmap(
            fwd,
            in_axes = (None, None, 0, None, None),
            out_axes = (0, None)
        ),
        static_argnames="inference_mode"
    )

    activations, f_rng = fwd_jit(
        *f_rng.partition(is_trainable),
        input,
        rng,
        False # inference_mode
    )
    assert lax.eq(
        activations[0],
        expected_train_output
    ).all()
    assert lax.eq(
        activations[1],
        jnp.stack([rng, rng])
    ).all()
    

    activations, f_rng = fwd_jit(
        *f_rng.partition(is_trainable),
        input,
        rng,
        True # inference_mode
    )
    assert lax.eq(
        activations[0],
        expected_infer_output
    ).all()
    assert lax.eq(
        activations[1],
        jnp.stack([rng, rng])
    ).all()
