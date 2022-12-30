import jax
from jax import (
    tree_util
)

def assert_valid_pytree(trainables, non_trainables, hyperparams):
    hp, tr, ntr = tree_util.tree_map(
        lambda hp, tr, ntr: (hp, tr, ntr),
        hyperparams, trainables, non_trainables
    )
    
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