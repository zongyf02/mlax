import jax
from jax import (
    tree_util
)
from typing import Any
from dataclasses import dataclass

@dataclass(slots=True)
class Data:
    tr: Any
    ntr: Any
    hp: Any

def assert_valid_pytree(trainables, non_trainables, hyperparams):
    combined = tree_util.tree_map(
        lambda hp, tr, ntr: Data(tr, ntr, hp),
        hyperparams, trainables, non_trainables
    )
    combined_leaves, combined_tree_def = tree_util.tree_flatten(combined)
    
    tr_leaves = map(lambda data: data.tr, combined_leaves)
    ntr_leaves = map(lambda data: data.ntr, combined_leaves)
    hp_leaves = map(lambda data: data.hp, combined_leaves)

    tr = tree_util.tree_unflatten(combined_tree_def, tr_leaves)
    ntr = tree_util.tree_unflatten(combined_tree_def, ntr_leaves)
    hp = tree_util.tree_unflatten(combined_tree_def, hp_leaves)

    tr_eq = tr == trainables
    if isinstance(tr_eq, jax.Array):
        assert tr_eq.all()
    else:
        assert tr_eq

    ntr_eq = ntr == non_trainables
    if isinstance(ntr_eq, jax.Array):
        assert ntr_eq.all()
    else:
        assert ntr_eq

    assert hp == hyperparams
