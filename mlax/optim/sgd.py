from jax import (
    lax,
    tree_util
)
from typing import NamedTuple, Any

class State(NamedTuple):
    velocities: Any

def init(weights):
    """Initialize state for an SGD optimizer.

    :param weights: Pytree containing the weights for the SGD to optimize.

    :returns state: State whose velocities is a pytree of the same structure as
        ``weights``, where each leaf contains a zero initial velocity array of
        the same shape as the corresponding leaf in ``weights``.
    """
    velocities = tree_util.tree_map(
        lambda w: lax.full_like(w, 0, w.dtype),
        weights
    )
    return State(velocities)

def step(
    gradients,
    optim_state,
    lr = 0.01,
    momentum = 0.0,
    nesterov = False
):
    """Find the gradients for a single optimization step.

    :param gradients: Gradients to optimize on.
    :param optim_sate: State of optmizer. Must be resulted from ``init`` on
        ``weights`` of the same shape as ``gradients``.
    :param lr: Learning rate. Default: 0.01
    :param momentum: SGD momentum. Default: 0.0
    :param nesterov: Whether to use Nesterov momentum. Default: False.
    """
    def _step(
        gradients,
        velocity
    ):
        _lr = lax.convert_element_type(lr, gradients.dtype)
        _momentum = lax.convert_element_type(momentum, velocity.dtype)

        velocity = lax.add(
            lax.mul(velocity, _momentum),
            lax.mul(gradients, _lr)
        )

        if nesterov:
            gradients = lax.add(
                lax.mul(velocity, _momentum),
                lax.mul(gradients, _lr)
            ) 
        else:
            gradients = velocity

        return gradients, velocity
    
    gradients, velocities = tree_util.tree_transpose(
        outer_treedef = tree_util.tree_structure(gradients),
        inner_treedef = tree_util.tree_structure((0, 0)),
        pytree_to_transpose = tree_util.tree_map(
            lambda g, s: _step(g, s),
            gradients, optim_state.velocities
        )
    )

    return gradients, State(velocities)
    