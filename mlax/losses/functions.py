from jax import lax

def l2_loss(predictions, targets):
    """Calculates the l2 loss of a pytree of weights.

    :param predictions: Array of predictions.
    :param tragets: Array of targets, of the same shape as ``predictions``.
    
    :returns loss: L2 loss of ``predictions`` against ``targets``.
    """
    loss = lax.mul(weight, weights)
    loss = lax.reduce(loss, 0, lax.add, tuple(range(loss.ndim)))
    return loss

def categorical_crossentropy(predictions, targets):
    """Calculates the categorical crossentropy of predictions.

    :param predictions: Array of predicted probabilities.
    :param targets: Array of target probabilities, of the same shape as
        ``predictions``.

    :returns loss: Categorical cross-entropy loss of ``predictions`` against 
        ``targets``.
    """
    loss = lax.mul(lax.log(predictions), targets)
    loss = lax.reduce(loss, 0, lax.add, tuple(range(loss.ndim)))
    return lax.neg(loss)
