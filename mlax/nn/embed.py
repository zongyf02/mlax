from mlax import Parameter, Module
from jax import (
    Array,
    numpy as jnp,
    nn,
    dtypes
)
from typing import Any, Tuple, Union, Hashable

class Embed(Module):
    """Embedding layer."""
    def __init__(
        self,
        rng: Array,
        vocab_size: int,
        embed_dim: int,
        embed_initializer=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', in_axis=1, out_axis=0),
        dtype=jnp.float32
    ):
        """Initialize an embedding layer.
 
        :param rng: PRNG key.
        :vocab_size: Size of the vocabulary to embed.
        :embed_dim: Size of each embedding.
        :embed_inititializer: Initializer for embedding weight of shape
            ``(vocab_size, embed_dim)`` as defined by
            `jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>`_.
            Default: normal variance scaling.
        :param dtype: Type of initialized parameters. Default: float32.
        """
        super().__init__()

        self.rng = rng
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.embed_initializer = embed_initializer
        self.dtype = dtypes.canonicalize_dtype(dtype)

        self.embed_kernel = Parameter(trainable=True)

    def init(self, x: Array) -> None:
        self.embed_kernel.data=self.embed_initializer(
            self.rng, (self.vocab_size, self.embed_dim), self.dtype
        )

    def apply(
        self,
        x: Array,
        rng: None=None,
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> Tuple[Array, Any]:
        return self.embed_kernel.data.at[x].get(mode="fill")
