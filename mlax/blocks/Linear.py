from jax import nn
from jax import random
from typing import NamedTuple, Any

from mlax.nn import linear, bias

class Weights(NamedTuple):
    kernel: Any
    bias: Any

def init(
    key,
    in_features, 
    out_features,
    kernel_initializer=nn.initializers.glorot_uniform(),
    bias_initializer=nn.initializers.zeros,
    dtype="float32"
) -> Weights:
    """Intialize weights for a Linear block.

    :param key: PRNG key for weights initialization.
    :param in_features: Number of input features.
    :param out_features: Number of desired output features.
    :param kernel_initializer: Initializer as defined by
        `jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>`_.
        Default:: glorot uniform.
    :param bias_initializer: Initializer as defined by
        `jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>`_.
        Default:: zeros.
    :param dtype: Type of the weights. Default: float32.

    :returns weights: Initialized Weights.
    """
    linear_key, bias_key = random.split(key)
    return Weights(
        kernel = linear.init(
            linear_key, in_features, out_features, kernel_initializer, dtype
        ),
        bias = bias.init(bias_key, (out_features, ), bias_initializer, dtype)
    )

def fwd(
    x,
    weights: Weights,
    activation_fn=None,
    precision=None, 
):
    """Apply linear transformation with bias to input features.

    :param x: Input features to the Linear block. Must be of the shape of
        ``num_in_features`` or (n_batches, ``num_in_features``) and of the same
        type as elements of ``weights``.
    :param weights: Initialized weights for a Linear block.
    :param activation_fn: Activation functions of the signature ``Array -> Array``. See
        `jax.nn <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot.html#jax.lax.dot>`_.
        for examples.
        Default: None, no activation function is applied.
    :param precision: See ``precision`` parameter of
        `jax.lax.dot <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot.html#jax.lax.dot>`_,
        which is used internally in the Linear block.
        Default None.

    :returns y: Linear transformation with bias applied ``x``.
    """
    kernel_weights, bias_weights = weights
    x = bias.fwd(
        linear.fwd(x, kernel_weights, precision),
        bias_weights
    )

    if activation_fn is None:
        return x
    else:
        return activation_fn(x)
    