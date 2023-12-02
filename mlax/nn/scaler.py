from typing import Sequence, Optional, Union, Tuple, Hashable
from jax import (
    Array,
    numpy as jnp,
    nn,
    lax,
    dtypes
)
from mlax import Parameter, Variable, Module
from mlax._utils import _canon_int_sequence

class Scaler(Module):
    """Scaler layer."""
    def __init__(
        self,
        rng: Array,
        in_features: Union[int, Sequence[Optional[int]]],
        scaler_initializer=nn.initializers.ones,
        dtype=jnp.float32
    ):
        """Initialize a bias layer.

        :param rng: PRNG key.
        :param in_features: Integer or sequence of optional integers indicating
            the shape of the input features to scale. Empty sequence indicates a
            single scalar. For per-axis scaling, use ``0`` on axes that do not
            require scaling, use ``1`` on axes that require a single scaling
            term, and ``-1`` or ``axis_length`` on axes that require a scaling
            term for each of their ``axis_length`` elements. A single integer
            is interpreted as a sequence of one.
        :param scaler_initializer: Scaler initializer as defined by
            `jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>`_.
            Default: ones.
        :param dtype: Type of initialized parameters. Default: float32.
        """
        super().__init__()

        self.rng = Variable(data=rng)
        self.in_features = _canon_int_sequence(in_features, 1)
        self.scaler_initializer = scaler_initializer
        self.dtype = dtypes.canonicalize_dtype(dtype)

        self.scaler_kernel = Parameter()

    def set_up(self, x: Array) -> None:
        scaler_shape = [
            axis if axis != -1 else x.shape[i]
            for i, axis in enumerate(self.in_features) if axis != 0
        ]
        self.scaler_kernel.data=self.scaler_initializer(
            self.rng.data, scaler_shape, self.dtype
        )
    
    def forward(
        self,
        x: Array,
        rng: None=None,
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> Array:
        return lax.mul(
            x,
            lax.broadcast_in_dim(
                lax.convert_element_type(self.scaler_kernel.data, x.dtype),
                x.shape,
                [i for i, axis in enumerate(self.in_features) if axis != 0]
            )
        )
