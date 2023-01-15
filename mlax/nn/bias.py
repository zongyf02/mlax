from mlax import Parameter, Module
from jax import (
    numpy as jnp,
    nn,
    lax
)
from typing import Any, Sequence, Optional, Union
from mlax._utils import (
    _canon_int_sequence,
    _canon_dtype
)


class Bias(Module):
    """Bias addition layer."""
    def __init__(
        self,
        rng: Any,
        in_features: Union[int, Sequence[Optional[int]]],
        bias_initializer=nn.initializers.zeros,
        dtype=jnp.float32
    ):
        """Initialize a bias layer.

        :param rng: PRNG key for weight initialization.
        :param in_features: Integer or sequence of optional integers indicating
            the shape of the input features to add bias to. Empty sequence
            indicates a scalar bias. For non-scaler biases, use ``None`` on axes
            that do not require a bias, use ``1`` on axes that require a single
            bias term, and ``-1`` or ``axis_length`` on axes that require a bias
            term for each of their ``axis_length`` elements. A single integer
            is interpreted as a sequence of one.
        :param bias_initializer: Bias initializer as defined by
            `jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>`_.
            Default: zeros.
        :param dtype: Dtype of initialized bias weight. Default: float32.
        """
        super().__init__()
        self.initialized = False

        self._rng = Parameter(trainable=False, data=rng)
        self._in_features = _canon_int_sequence(in_features, 1)
        self._bias_initializer = bias_initializer
        self._dtype = _canon_dtype(dtype)

        self.bias_weight = Parameter(trainable=True)
        self.bias_broadcast_dims = None

    def _build(self, x):
        """Initialize an uninitialized bias layer."""
        self.bias_broadcast_dims = tuple(
            i for i, axis in enumerate(self._in_features)
            if axis is not None
        )

        bias_shape = [
            axis if axis != -1 else x.shape[axis]
            for axis in self._in_features if axis is not None
        ]
        self.bias_weight.data = self._bias_initializer(
            self._rng.data,
            bias_shape,
            self._dtype
        )

        del self._rng
        del self._in_features
        del self._bias_initializer
        del self._dtype

        self.initialized = True
    
    def __call__(self, x, rng=None, inference_mode=False):
        """Add bias to input features.
        
        :param self: Bias layer.
        :param x: Input features to the bias layer. Must be of the shape
            ``in_feature_shape``.
        :param rng: PRNG key. Ignored. Default: None.
        :param inference_mode: Whether in inference or training mode. Ignored.
            Default: False.
        
        :returns y: ``x`` plus bias.
        :returns: Bias layer with updated state. Possibly the same object as
            ``self``.
        """
        if not self.initialized:
            self._build(x)

        return lax.add(
            x,
            lax.broadcast_in_dim(
                lax.convert_element_type(self.bias_weight.data, x.dtype),
                x.shape,
                self.bias_broadcast_dims
            )
        ), self
