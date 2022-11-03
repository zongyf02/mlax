from jax import lax

def categorical_crossentropy(predictions, targets):
    """Calculates the categorical crossentropy of predictions.

    :param predictions: Array of predicted probabilities.
    :param targets: Array of target probabilities, of the same shape as
        ``predictions``.

    :returns loss: Array of per-label categorical cross-entropy loss of
        ``predictions`` against ``targets``, same shape as ``predictions``.
    """
    loss = lax.mul(lax.log(predictions), targets)
    return lax.neg(loss)
