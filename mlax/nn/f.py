import jax
from mlax import Module
from typing import Callable, Optional, Any

class F(Module):
    """Wrapper to create pure function layers."""
    def __init__(
        self,
        train_fn: Callable[[jax.Array], jax.Array],
        infer_fn: Optional[Callable[[jax.Array], jax.Array]] = None
    ):
        """Initialize a F layer.

        :param train_fn: Pure function that takes in and returns a JAX array.
            Called when ``inference_mode`` is False.
        :param infer_fn: Optional pure function that takes in and returns a JAX
            array. Called when ``inference_mode`` is True. If None, the
            ``train_fn`` is called instead. Default: None.
        """
        super().__init__()
        self.train_fn = train_fn
        self.infer_fn = infer_fn
    
    def __call__(self, x, rng=None, inference_mode=False):
        """Apply an arbitrary pure functional transform.
        
        :param self: F layer.
        :param x: Input features.
        :param rng: PRNG key. Ignored. Default: None.
        :param inference_mode: Whether in inference or training mode. Default:
            False.
        
        :returns: ``x`` with the arbitrary transform applied.
        :returns: F layer with updated state. Possibly the same object as
            ``self``.
        """
        if inference_mode:
            if self.infer_fn is None:
                return self.train_fn(x), self
            else:
                return self.infer_fn(x), self
        else:
            return self.train_fn(x), self

class FRng(Module):
    """Wrapper to create pure function layers that may require rng."""
    def __init__(
        self,
        train_fn: Callable[[jax.Array, Any], jax.Array],
        infer_fn: Optional[Callable[[jax.Array, Any], jax.Array]] = None
    ):
        """Initialize a FRng layer.

        :param train_fn: Pure function that takes in a JAX array and a PRNGKey
            and returns a JAX array. Called when ``inference_mode`` is False.
        :param infer_fn: Optional pure function that takes in a JAX array and a
            PRNGKey and and returns a JAX array. Called when ``inference_mode``
            is True. If None, the ``train_fn`` is called instead. Default: None.
        """
        self.train_fn = train_fn
        self.infer_fn = infer_fn
    
    def __call__(self, x, rng, inference_mode=False):
        """Apply an arbitrary pure functional transform.
        
        :param self: FRng layer.
        :param x: Input features.
        :param rng: PRNG key.
        :param inference_mode: Whether in inference or training mode. Default:
            False.
        
        :returns: ``x`` with the arbitrary transform applied.
        :returns: F layer with updated state. Possibly the same object as
            ``self``.
        """
        if inference_mode:
            if self.infer_fn is None:
                return self.train_fn(x, rng), self
            else:
                return self.infer_fn(x, rng), self
        else:
            return self.train_fn(x, rng), self
