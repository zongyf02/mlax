from mlax import Parameter, Module
from jax import (
    Array,
    numpy as jnp,
    nn,
    lax,
    random,
    dtypes
)
from typing import Any, Union, Sequence, Hashable, Tuple
from mlax._utils import (
    _canon_int_sequence,
    _compute_std_stats,
    _standadize
)

class ZNorm(Module):
    """Z-score normalization across batch axes with running mean and variance."""
    def __init__(
        self,
        rng: Array,
        axis: Union[str, int, Sequence[int]],
        epsilon: float=1e-05,
        momentum: float=0.9,
        mean_initializer=nn.initializers.zeros,
        variance_initializer=nn.initializers.ones,
        dtype=jnp.float32
    ):
        """Initialize a normalization layer.

        :param rng: PRNG key.
        :param axis: "channel_last", "channel_first", axis, or sequence of axes
            to normalize input features along. "channel_last" and
            "channel_first" indicate normalization along all but the channel
            axis, assumed to be the last or first axis (batch norm).
        :param epsilon: Small number added to variance to avoid divisions by
            zero. Default: 1e-05.
        :param momentum: Momentum for the moving average. Default: 0.9.
        :param mean_initializer: Initializer for moving mean of shape
            ``(n_channels,)`` as defined by
            ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
            Default:: zeros.
        :param variance_initializer: Initializer for moving variance of shape
            ``(n_channels,)`` as defined by
            ``jax.nn.initalizers <https://jax.readthedocs.io/en/latest/jax.nn.initializers.html>``.
            Default:: ones.
        :param dtype: Type of initialized parameters. Default: float32.
        """
        super().__init__()

        self.rng = rng
        self.axis = (
            str(axis) if isinstance(axis, str) else _canon_int_sequence(axis, 1)
        )
        self.epsilon = float(epsilon)
        self.momentum = float(momentum)
        self.mean_initializer = mean_initializer
        self.variance_initializer = variance_initializer
        self.dtype = dtypes.canonicalize_dtype(dtype)
    
        self.moving_mean = Parameter(trainable=False)
        self.moving_var = Parameter(trainable=False)

    def init(self, x: Array) -> None:
        if self.axis == "channel_last":
            shape = x.shape[-1]
        elif self.axis == "channel_first":
            shape = x.shape[0]
        else:
            shape = [x.shape[i] for i in range(x.ndim) if i not in self.axis]

        self.moving_mean.data = self.mean_initializer(
            random.fold_in(self.rng, 0), shape, self.dtype
        )
        self.moving_var.data = self.variance_initializer(
            random.fold_in(self.rng, 1), shape, self.dtype
        )

    def apply(
        self,
        x: Array,
        rng: None=None,
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> Tuple[Array, Any]:
        """Apply normalization to input features."""
        if self.axis == "channel_last":
            axis = list(range(x.ndim - 1))
        elif self.axis == "channel_first":
            axis = list(range(1, x.ndim))
        else:
            axis = self.axis

        if inference_mode is True:
            mean = lax.convert_element_type(self.moving_mean.data, x.dtype)
            variance = lax.convert_element_type(self.moving_var.data, x.dtype)
        else:
            mean, variance = _compute_std_stats(x, axis, batch_axis_name)

            # Update running stats
            one_m_momemtum = 1.0 - self.momentum
            self.moving_mean.data = lax.convert_element_type(lax.add(
                lax.mul(
                    lax.convert_element_type(self.moving_mean.data, x.dtype),
                    lax.convert_element_type(self.momentum, x.dtype)
                ),
                lax.mul(mean, lax.convert_element_type(one_m_momemtum, x.dtype))
            ), self.moving_mean.data.dtype)
            self.moving_var.data = lax.convert_element_type(lax.add(
                lax.mul(
                    lax.convert_element_type(self.moving_var.data, x.dtype),
                    lax.convert_element_type(self.momentum, x.dtype)
                ),
                lax.mul(
                    variance, lax.convert_element_type(one_m_momemtum, x.dtype)
                )
            ), self.moving_var.data.dtype)

        return _standadize(x, axis, mean, variance, self.epsilon)
