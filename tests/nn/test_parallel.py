import jax
from jax import (
    numpy as jnp,
    tree_util as jtu,
    random,
    nn,
    lax
)
from mlax import (
    fwd,
    is_trainable
)
from mlax.nn import Parallel, ParallelRng, Scaler, F, FRng
import pytest

@pytest.mark.parametrize(
    "layers,input,expected_train_output,expected_infer_output",
    [
        (
            (
                Scaler(
                    random.PRNGKey(0),
                    in_features=-1, 
                    scaler_initializer=nn.initializers.constant(2)
                ),
                F(
                    train_fn=lambda x: x,
                    infer_fn=lambda x: 2 * x
                ),
                F(
                    train_fn=lambda x: 3 * x
                )
            ),
            [
                jnp.ones((2, 4), jnp.bfloat16),
                jnp.ones((2, 3), jnp.bfloat16),
                jnp.ones((2, 2), jnp.float32)
            ],
            [
                jnp.full((2, 4), 2, jnp.bfloat16),
                jnp.ones((2, 3), jnp.bfloat16),
                jnp.full((2, 2), 3, jnp.float32)
            ],
            [
                jnp.full((2, 4), 2, jnp.bfloat16),
                jnp.full((2, 3), 2, jnp.bfloat16),
                jnp.full((2, 2), 3, jnp.float32)
            ]
        ),
    ]
)
def test_parallel(
    layers,
    input,
    expected_train_output,
    expected_infer_output,
):
    model = Parallel(
        layers
    )

    fwd_jit = jax.jit(
        jax.vmap(
            fwd,
            in_axes = (None, None, 0, None, None),
            out_axes = (0, None)
        ),
        static_argnames="inference_mode"
    )

    activations, model = fwd_jit(
        *model.partition(),
        input,
        None, # rng
        False # inference_mode
    )
    assert jtu.tree_reduce(
        lambda acc, x: acc and x,
        jtu.tree_map(
            lambda a, b: lax.eq(a, b).all(),
            activations,
            expected_train_output
        )
    )
    
    activations, model = fwd_jit(
        *model.partition(),
        input,
        None, # rng
        True # inference_mode
    )
    assert jtu.tree_reduce(
        lambda acc, x: acc and x,
        jtu.tree_map(
            lambda a, b: lax.eq(a, b).all(),
            activations,
            expected_infer_output
        )
    )

@pytest.mark.parametrize(
    "layers,input,rng,expected_train_output,expected_infer_output",
    [
        (
            iter((
                Scaler(
                    random.PRNGKey(0),
                    in_features=-1, 
                    scaler_initializer=nn.initializers.constant(2)
                ),
                F(
                    train_fn=lambda x: x,
                    infer_fn=lambda x: 2 * x
                ),
                FRng(
                    train_fn=lambda x, rng: (3 * x, rng)
                )
            )),
            [
                jnp.ones((2, 4), jnp.bfloat16),
                jnp.ones((2, 3), jnp.bfloat16),
                jnp.ones((2, 2), jnp.float32)
            ],
            random.PRNGKey(1),
            [
                jnp.full((2, 4), 2, jnp.bfloat16),
                jnp.ones((2, 3), jnp.bfloat16),
                (
                    jnp.full((2, 2), 3, jnp.float32),
                    jnp.stack([random.PRNGKey(1), random.PRNGKey(1)])
                )
            ],
            [
                jnp.full((2, 4), 2, jnp.bfloat16),
                jnp.full((2, 3), 2, jnp.bfloat16),
                (
                    jnp.full((2, 2), 3, jnp.float32),
                    jnp.stack([random.PRNGKey(1), random.PRNGKey(1)])
                )
            ]
        ),
    ]
)
def test_parallel_rng(
    layers,
    input,
    rng,
    expected_train_output,
    expected_infer_output,
):
    model = ParallelRng(
        layers
    )

    fwd_jit = jax.jit(
        jax.vmap(
            fwd,
            in_axes = (None, None, 0, None, None),
            out_axes = (0, None)
        ),
        static_argnames="inference_mode"
    )

    activations, model = fwd_jit(
        *model.partition(),
        input,
        rng, # rng
        False # inference_mode
    )
    assert jtu.tree_reduce(
        lambda acc, x: acc and x,
        jtu.tree_map(
            lambda a, b: lax.eq(a, b).all(),
            activations,
            expected_train_output
        )
    )

    activations, model = fwd_jit(
        *model.partition(),
        input,
        rng, # rng
        True # inference_mode
    )
    assert jtu.tree_reduce(
        lambda acc, x: acc and x,
        jtu.tree_map(
            lambda a, b: lax.eq(a, b).all(),
            activations,
            expected_infer_output
        )
    )
