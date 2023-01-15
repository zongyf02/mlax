from mlax import Parameter, Module
from jax import (
    numpy as jnp,
    nn,
    lax
)
from typing import Any
from mlax._utils import (
    _canon_dtype,
    _canon_opt_dtype,
    _canon_precision_pair
)


class Linear(Module):
    """Linear transformation layer without bias with lazy kernel initialization."""
    def __init__(
        self,
        rng: Any,
        out_features: int,
        precision=None,
        accum_dtype=None,
        transposed_kernel=False,
        kernel_initializer=nn.initializers.glorot_uniform(),
        dtype=jnp.float32
    ):
        """Initialize a linear layer.

        :param rng: PRNG key for weight initialization.
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
        :param kernel_initializer: Kernel initializer as defined by
            `jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>`_.
            Default: glorot uniform.
        :param dtype: Dtype of initialized kernel weight. Default: float32.
        """
        super().__init__()
        self.initialized = False

        self._rng =  Parameter(trainable=False, data=rng)
        self._out_features = int(out_features)
        self._kernel_initializer = kernel_initializer
        self._dtype = _canon_dtype(dtype)
        
        self.transposed_kernel = bool(transposed_kernel)
        self.kernel_weight = Parameter(trainable=True)
        self.precision = _canon_precision_pair(precision)
        self.accum_dtype = _canon_opt_dtype(accum_dtype)

    def _build(self, x):
        """Initialize an uninitialized linear layer."""
        self.kernel_weight.data = self._kernel_initializer(
            self._rng.data,
            (x.shape[-1], self._out_features),
            self._dtype
        )

        del self._rng      
        del self._out_features
        del self._kernel_initializer
        del self._dtype

        if self.transposed_kernel:
            self.kernel_weight.data = lax.transpose(
                self.kernel_weight.data,
                (1, 0)
            )
        
        self.initialized = True
    
    def __call__(self, x, rng=None, inference_mode=False):
        """Apply linear transformation to input features.
        
        :param self: Linear layer.
        :param x: Input features to the linear layer. Must be of the shape
            ``(..., in_features,)``.
        :param rng: PRNG key. Ignored. Default: None.
        :param inference_mode: Whether in inference or training mode. Ignored.
            Default: False.
        
        :returns: ``x`` with linear transformation applied. Shape
            ``(..., out_features)``.
        :returns: Linear layer with updated state. Possibly the same object as
            ``self``.
        """
        if not self.initialized:
            self._build(x)

        contracting_dims = (1,) if self.transposed_kernel else (0,)
        return lax.dot_general(
            x,
            lax.convert_element_type(self.kernel_weight.data, x.dtype),
            (((x.ndim - 1,), contracting_dims), ((), ())),
            self.precision,
            self.accum_dtype
        ), self
