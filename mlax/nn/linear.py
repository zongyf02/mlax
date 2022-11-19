from jax import (
    nn,
    lax
)

def init(
    key,
    in_features, 
    out_features,
    kernel_initializer=nn.initializers.glorot_uniform(),
    dtype=None
):
    """Intialize weights for a linear transform.

    :param key: PRNG key for weight initialization.
    :param in_features: Number of input features.
    :param out_features: Number of desired output features.
    :param kernel_initializer: Initializer as defined by
        ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
        Default:: glorot uniform.
    :param dtype: Type of initialized weights. Default: None, which is the
        ``kernel_initializer``'s default.

    :returns weight: Initialized kernel weight of shape
        ``(in_features, out_features)``.
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
        ``(n_batches, in_features)``.
    :param weights: Initialized kernel weight for a linear transform.
    :param precision: See ``precision`` parameter of
        ``jax.lax.dot <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot.html#jax.lax.dot>``.
        Default None.
    :param preferred_element_type: See ``preferred_element_type`` parameter of
        ``jax.lax.dot <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot.html#jax.lax.dot>``.
        Default None.

    :returns y: ``x`` dot ``weights``.
    """
    return lax.dot_general(
        x,
        weights,
        (((1,), (0,)), ((), ())),
        precision,
        preferred_element_type
    )
