from jax import (
    nn,
    lax
)

def init(
    key,
    features_shape,
    bias_initializer=nn.initializers.zeros,
    dtype="float32"
):
    """Intialize weights for a bias transform.

    :param key: PRNG key for weight initialization.
    :param features_shape: Shape of the features to add bias to.
    :param bias_initializer: Initializer as defined by
        ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
        Default:: zeros.
    :param dtype: Type of initialized weights. Default: float32.

    :returns weight: Initialized bias weight of ``features_shape``.
    """
    return bias_initializer(key, features_shape, dtype)

def fwd(x, weights):
    """Add bias to input features.

    :param x: Input features to the bias transform. Must be of the same shape as
        ``weights`` or (n_batches, ``weights``).
    :param weights: Initialized bias weight for a bias transform.

    :returns y: ``x`` plus bias weight.
    """
    if x.shape[1:] == weights.shape:
        return lax.add(x, lax.broadcast(weights, (x.shape[0],)))
    else:
        return lax.add(x, weights)
