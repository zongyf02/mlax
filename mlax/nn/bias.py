from jax import (
    Array,
    numpy as jnp,
    nn,
    lax,
    dtypes
)
from typing import Sequence, Union, Tuple, Hashable, Any
from mlax import Parameter, Module
from mlax._utils import _canon_int_sequence

class Bias(Module):
    """Bias addition layer."""
    def __init__(
        self,
        rng: Array,
        in_features: Union[int, Sequence[int]],
        bias_initializer=nn.initializers.zeros,
        dtype=jnp.float32
    ):
        """Initialize a bias layer.

        :param rng: PRNG key.
        :param in_features: Integer or sequence of integers indicating the shape
            of the input features to add bias to. Empty sequence indicates a
            scalar bias. For non-scaler biases, use ``0`` on axes that do not
            require a bias, use ``1`` on axes that require a single bias term,
            and ``-1`` or ``axis_length`` on axes that require a bias term for
            each of their ``axis_length`` elements. A single integer is
            interpreted as a sequence of one.
        :param bias_initializer: Bias initializer as defined by
            `jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>`_.
            Default: zeros.
        :param dtype: Type of initialized parameters. Default: float32.
        """
        super().__init__()

        self.rng = rng
        self.in_features = _canon_int_sequence(in_features, 1)
        self.bias_initializer = bias_initializer
        self.dtype = dtypes.canonicalize_dtype(dtype)

        self.bias_kernel = Parameter(trainable=True)

    def init(self, x: Array) -> None:
        bias_shape = [
            axis if axis != -1 else x.shape[i]
            for i, axis in enumerate(self.in_features) if axis != 0
        ]
        self.bias_kernel.data = self.bias_initializer(
            self.rng, bias_shape, self.dtype
        )

    def apply(
        self,
        x: Array,
        rng: None=None,
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> Tuple[Array, Any]:
        """Add bias to input features."""
        return lax.add(
            x,
            lax.broadcast_in_dim(
                lax.convert_element_type(self.bias_kernel.data, x.dtype),
                x.shape,
                [i for i, axis in enumerate(self.in_features) if axis != 0]
            )
        )
