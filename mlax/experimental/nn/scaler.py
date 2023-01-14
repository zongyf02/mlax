from mlax.experimental import Parameter, Module
from jax import (
    numpy as jnp,
    nn,
    lax
)
from typing import Any, Sequence, Optional
from mlax.experimental._utils import (
    _canon_dtype
)

class Scaler(Module):
    """Scaler layer."""
    def __init__(
        self,
        rng: Any,
        in_feature_shape: Sequence[Optional[int]],
        scaler_initializer=nn.initializers.ones,
        dtype=jnp.float32
    ):
        """Initialize a bias layer.

        :param rng: PRNG key for weight initialization.
        :param in_feature_shape: Shape of the input features to scale. Empty
            sequence indicates a single scalar. For per-axis scaling, use
            ``None`` on axes that do not require scaling, use ``1`` on axes that
            require a single scaling term, and ``-1`` or ``axis_length`` on axes
            that require a scaling term for each of their ``axis_length``
            elements.
        :param scaler_initializer: Scaler initializer as defined by
            `jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>`_.
            Default: ones.
        :param dtype: Dtype of initialized bias weight. Default: float32.
        """
        super().__init__()
        self.initialized = False

        self._rng = Parameter(trainable=True, data=rng)
        self._in_feature_shape = tuple(in_feature_shape)
        self._scaler_initializer = scaler_initializer
        self._dtype = _canon_dtype(dtype)

        self.scaler_weight = Parameter(trainable=True)
        self.scaler_broadcast_dims = None

    def _build(self, x):
        """Initialize an uninitialized scaler layer."""
        self.scaler_broadcast_dims = tuple(
            i for i, axis in enumerate(self._in_feature_shape)
            if axis is not None
        )

        scaler_shape = [
            axis if axis != -1 else x.shape[axis]
            for axis in self._in_feature_shape if axis is not None
        ]
        self.scaler_weight.data = self._scaler_initializer(
            self._rng.data,
            scaler_shape,
            self._dtype
        )

        del self._rng
        del self._in_feature_shape
        del self._scaler_initializer
        del self._dtype

        self.initialized = True
    
    def __call__(self, x, rng=None, inference_mode=False):
        """Scale input features.
        
        :param self: Scaler layer.
        :param x: Input features to the scaler layer. Must be of the shape
            ``in_feature_shape``.
        :param rng: PRNG key. Ignored. Default: None.
        :param inference_mode: Whether in inference or training mode. Ignored.
            Default: False.
        
        :returns y: Scaled ``x``.
        :returns: Bias layer with updated state. Possibly the same object as
            ``self``.
        """
        if not self.initialized:
            self._build(x)

        return lax.mul(
            x,
            lax.broadcast_in_dim(
                lax.convert_element_type(self.scaler_weight.data, x.dtype),
                x.shape,
                self.scaler_broadcast_dims
            )
        ), self
