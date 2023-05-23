from jax import (
    Array,
    numpy as jnp,
    nn,
    lax,
    dtypes
)
from typing import Any, Tuple, Union, Hashable
from mlax import Parameter, Module
from mlax._utils import (
    _canon_opt_dtype,
    _canon_precision_pair
)

class Linear(Module):
    """Linear transformation layer without bias with lazy kernel initialization."""
    def __init__(
        self,
        rng: Array,
        out_features: int,
        precision=None,
        accum_dtype=None,
        transposed_kernel: bool=False,
        kernel_initializer=nn.initializers.glorot_uniform(),
        dtype=jnp.float32
    ):
        """Initialize a linear layer.

        :param rng: PRNG key.
        :param out_features: Number of output features.
        :param precision: See ``precision`` parameter of
            `jax.lax.dot <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot_general.html>`_,
            which is used internally in the forward pass. Default: None.
        :param accum_dtype: See ``preferred_element_type`` parameter of
            `jax.lax.dot <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot_general.html>`_,
            which is used internally in the forward pass. Default: None.
        :param transposed_kernel: Whether the kernel is of the shape
            ``(out_features, in_features)`` or ``(in_features, out_features)``.
            Default: False, the former.
        :param kernel_initializer: Initializer for kernel of shape
            ``(in_features, out_features)`` as defined by
            `jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>`_.
            Default: glorot uniform.
        :param dtype: Type of initialized parameters. Default: float32.
        """
        super().__init__()

        self.rng = rng
        self.out_features = int(out_features)
        self.precision = _canon_precision_pair(precision)
        self.accum_dtype = _canon_opt_dtype(accum_dtype)
        self.kernel_initializer = kernel_initializer
        self.transposed_kernel = transposed_kernel
        self.dtype = dtypes.canonicalize_dtype(dtype)

        self.linear_kernel = Parameter(trainable=True)

    def init(self, x: Array) -> None:
        """Initialize an uninitialized linear layer."""
        self.linear_kernel.data = self.kernel_initializer(
            self.rng, (x.shape[-1], self.out_features), self.dtype
        )
        if self.transposed_kernel:
            self.linear_kernel.data = lax.transpose(
                self.linear_kernel.data, (1, 0)
            )

    def apply(
        self,
        x: Array,
        rng: None=None,
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> Tuple[Array, Any]:
        """Apply linear transformation to input features."""
        contracting_dims = (1,) if self.transposed_kernel else (0,)
        return lax.dot_general(
            x,
            lax.convert_element_type(self.linear_kernel.data, x.dtype),
            (((x.ndim - 1,), contracting_dims), ((), ())),
            self.precision,
            self.accum_dtype
        )
