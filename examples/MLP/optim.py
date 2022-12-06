from jax import (
    lax,
    tree_util,
    vmap
)
from typing import NamedTuple, Any

def sparse_categorical_crossentropy(predictions, targets):
    """Calculates the sparse categorical crossentropy of predictions.
    :param predictions: Array of predicted probabilities of the shape
        ``(batch_size, classes)``.
    :param targets: Array of indices indicating the correct prediction. Must be
        of the shape ``(batch_size,)``.
    :returns loss: Mean batch sparse categorical cross-entropy loss.
    """
    def per_example_loss_fn(pred, target):
        return lax.neg(lax.log(pred)[target])

    # Clip zeros to avoid NaNs
    predictions = lax.clamp(
        lax.convert_element_type(1e-7, predictions.dtype),
        predictions,
        lax.convert_element_type(1 - 1e-7, predictions.dtype),
    )
    per_example_loss = vmap(per_example_loss_fn)(predictions, targets)
    return per_example_loss.mean()

class State(NamedTuple):
    velocities: Any

def sgd_init(weights):
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

def sgd_step(
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

def apply_updates(
    update_gradients,
    model_weights,
    minimize = True
):
    """Apply update gradients to weights.
    :param update_gradients: Pytree of update gradients. Must be of the same
        tree structure as ``model_weights``.
    :param model_weights: Pytree of model weights.
    :param minimize: Whether to minimize or maximize loss.
        Default: True (minimize).
    :returns new_weights: New weights with the same pytree structure as
        ``model_weights``. ``model_weights`` with udpate gradients applied.
    """
    if minimize:
        fn = lambda g, w: lax.sub(w, g)
    else:
        fn = lambda g, w: lax.add(w, g)
    return tree_util.tree_map(fn, update_gradients, model_weights)
