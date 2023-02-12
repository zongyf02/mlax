from mlax import Parameter, Module
from jax import (
    numpy as jnp,
    nn
)

class Embed(Module):
    """Embedding layer."""
    def __init__(
        self,
        rng,
        vocab_size: int,
        embed_dim: int,
        embed_initializer=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', in_axis=1, out_axis=0),
        dtype=jnp.float32
    ):
        """Initialize an embedding layer.
 
        :param rng: PRNG key for weight initialization.
        :vocab_size: Size of the vocabulary to embed.
        :embed_dim: Size of each embedding.
        :embed_inititializer: Initializer for embedding weight of shape
            ``(vocab_size, embed_dim)`` as defined by
            `jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>`_.
            Default: normal variance scaling.
        :param dtype: Dtype of initialized embedding weight. Default: float32.
        """
        super().__init__()
        self.embed_weight = Parameter(
            trainable=True,
            data=embed_initializer(
                rng,
                (vocab_size, embed_dim),
                dtype
            )
        )
    
    def __call__(self, x, rng=None, inference_mode=False):
        """Convert sequence of tokens into embeddings.

        :param x: Input tokens. Must be unbatched and of the shape
            ``(sequence_length,)``. Must be of an integer-like dtype.
        :param rng: PRNG key. Ignored. Default: None.
        :param inference_mode: Whether in inference or training mode. Ignored.
            Default: False.

        :returns: Tokens converted to embeddings. Out-of-bound tokens are
            converted to NaNs for inexact types and minimum values for exact
            types.
        :returns: Embed layer with updated state. Possibly the same object as
            ``self``.    
        """
        return self.embed_weight.data.at[x].get(mode="fill"), self
