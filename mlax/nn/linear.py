from jax import (
    nn,
    lax
)

def init(
    key,
    in_features, 
    out_features,
    kernel_initializer=nn.initializers.glorot_uniform(),
    dtype="float32"
):
    """Intialize weights for a linear transform.

    :param key: PRNG key for weight initialization.
    :param in_features: Number of input features.
    :param out_features: Number of desired output features.
    :param kernel_initializer: Initializer as defined by
        ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
        Default:: glorot uniform.
    :param dtype: Type of initialized weights. Default: float32.

    :returns weight: Initialized kernel weight of shape
        ``(in_feautres, out_features)``.
    """
    return kernel_initializer(key, (in_features, out_features), dtype)

def fwd(
    x,
    weights,
    precision=None,
    preferred_element_type=None
):
    """Apply linear transformation (without bias) to input features.

    :param x: Input features to the linear transform. Must be of the shape
        ``num_in_features`` or (n_batches, ``num_in_features``).
    :param weights: Initialized kernel weight for a linear transform.
    :param precision: See ``precision`` parameter of
        ``jax.lax.dot <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot.html#jax.lax.dot>``.
        Default None.
    :param preferred_element_type: See ``preferred_element_type`` parameter of
        ``jax.lax.dot <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot.html#jax.lax.dot>``.
        Default None.

    :returns y: ``x`` dot kernel weight.
    """
    return lax.dot(x, weights, precision, preferred_element_type)
