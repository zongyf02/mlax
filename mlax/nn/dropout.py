from jax import (
    random,
    lax
)

def fwd(x, key, prob, train=True):
    """Apply random dropouts to input features. Randomly sets some elements of
    the input features to 0.

    .. note:
        Dropouts are different even in a same batch.

    :param x: Input features to the dropout transform.
    :param key: PRNG key for randomizing dropouts.
    :param prob: Probability at which each element is dropped. Must be of a
        non-zero floating point type.
    :param train: Whether in training or inference mode. When `False`, dropouts
        are not applied. Default: True

    :returns y: x with dropouts applied (sparsified).
    """

    if train:
        mask = random.bernoulli(key, prob, x.shape)
        zeros = lax.full_like(x, 0)
        return lax.select(
            mask,
            lax.div(x, lax.convert_element_type(prob, x.dtype)),
            zeros
        )
    else:
        return x
