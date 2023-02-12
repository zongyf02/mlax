from mlax import Parameter, Module
from jax import (
    numpy as jnp,
    nn,
    lax,
    random
)
from typing import Any, Union, Sequence, Hashable
from mlax._utils import (
    _canon_dtype,
    _n_elems,
    _mean,
    _variance,
    _normalize
)


class BatchNorm(Module):
    """BatchNorm with lazy running stats initialization."""
    def __init__(
        self,
        rng: Any,
        batch_axis_name: Union[Hashable, Sequence[Hashable]],
        epsilon: float = 1e-05,
        momentum: float = 0.9,
        channel_last: bool=False,
        mean_initializer=nn.initializers.zeros,
        var_initializer=nn.initializers.ones,
        dtype=jnp.float32
    ):
        """Initialize a batch norm layer.

        :param rng: PRNG key for weight initialization.
        :param batch_axis_name: Hashable or sequence of hashables representing
            the batch axis name(s) over which batch norm is being run.
        :param epsilon: Small number added to variance to avoid divisions by
            zero. Default: 1e-05.
        :param momentum: Momentum for the moving average. Default: 0.9.
        :param channel_last: Whether features are channel-last or first.
            Default: False, channel-first.
        :param mean_initializer: Initializer for moving mean of shape
            ``(n_channels,)`` as defined by
            ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
            Default:: zeros.
        :param var_initializer: Initializer for moving variance of shape
            ``(n_channels,)`` as defined by
            ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
            Default:: ones.
        :param dtype: Type of initialized moving mean and variance weight.
            Default: float32.
        """
        super().__init__()
        self.initialized = False

        self._rng = Parameter(trainable=False, data=rng)
        self._mean_initializer = mean_initializer
        self._var_initializer = var_initializer
        self._dtype = _canon_dtype(dtype)
        
        self.batch_axis_name = (
            batch_axis_name if isinstance(batch_axis_name, Hashable) else
            tuple(batch_axis_name)
        )
        self.moving_mean = Parameter(trainable=False)
        self.moving_var = Parameter(trainable=False)
        self.epsilon = epsilon
        self.momentum = momentum
        self.channel_last = channel_last
        

    def _build(self, x):
        """Initialize an uninitialized batch norm layer."""
        key1, key2 = random.split(self._rng.data)
        shape = (x.shape[-1] if self.channel_last else x.shape[0],)
        self.moving_mean.data = self._mean_initializer(key1, shape, self._dtype)
        self.moving_var.data =  self._var_initializer(key2, shape, self._dtype)

        del self._rng      
        del self._mean_initializer
        del self._var_initializer
        del self._dtype
        
        self.initialized = True
    
    def __call__(self, x, rng=None, inference_mode=False):
        """Apply batch normalization to input features.

        :param x: Input features. Must be compatible with ``channel_last``.
        :param rng: PRNG key. Ignored. Default: None.
        :param inference_mode: Whether in inference or training mode. Ignored.
            Default: False.
        
        :returns: Batch normalized ``x``.
        :returns: BatchNorm layer with updated state. Possibly the same object
            as ``self``.
        """
        if not self.initialized:
            self._build(x)

        dims = range(x.ndim - 1) if self.channel_last else range(1, x.ndim)
        if inference_mode:
            mean = lax.convert_element_type(self.moving_mean.data, x.dtype)
            var = lax.convert_element_type(self.moving_var.data, x.dtype)
        else:
            n_elems = _n_elems(x, dims)
            mean = lax.pmean(_mean(x, dims, n_elems), self.batch_axis_name)
            mean_of_squares = lax.pmean(
                _mean(lax.integer_pow(x, 2), dims, n_elems),
                self.batch_axis_name
            )
            var = _variance(x, dims, mean, mean_of_squares)

            # Update running stats
            moving_dtype = self.moving_mean.data.dtype
            momentum = lax.convert_element_type(
                self.momentum, moving_dtype
            )
            one_m_momentum = lax.convert_element_type(
                1.0 - self.momentum, moving_dtype
            )
            self.moving_mean.data = lax.add(
                self.moving_mean.data * momentum,
                lax.convert_element_type(mean, moving_dtype) * one_m_momentum
            )
            self.moving_var.data = lax.add(
                self.moving_var.data * momentum,
                lax.convert_element_type(var, moving_dtype) * one_m_momentum
            )
        
        return _normalize(
            x, dims, self.epsilon, mean, var
        ), self

