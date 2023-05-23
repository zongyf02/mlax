from jax import Array
from typing import Any, Callable, Optional, Tuple, Union, Hashable
from mlax import Module
from mlax._utils import _needs_axis_name

class F(Module):
    """Wrapper to create pure function layers."""
    def __init__(
        self,
        train_fn: Union[Callable[[Any], Any], Callable[[Any], Any]],
        infer_fn: Optional[Union[Callable[[Any], Any], Callable[[Any], Any]]]=None
    ):
        """Initialize a F layer.

        :param train_fn: Pure function that takes in a valid JAX type and
            optionally a keyword argument `axis_name` and returns a valid JAX
            type. Called when ``inference_mode`` is False.
        :param infer_fn: Pure function that takes in a valid JAX type and
            optionally a keyword argument `axis_name` and returns a valid JAX
            type. Called when ``inference_mode`` is True. If None, the
            ``train_fn`` is called instead. Default: None.
        """
        super().__init__()
        self.train_fn = train_fn
        self.infer_fn = infer_fn
    
    def init(self, x: Any) -> None:
        pass

    def apply(
        self,
        x: Any,
        rng: None=None,
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> Tuple[Any, Any]:
        if self.infer_fn is None or not inference_mode:
            if _needs_axis_name(self.train_fn):
                return self.train_fn(x, axis_name=batch_axis_name)
            else:
                return self.train_fn(x)
        else:
            if _needs_axis_name(self.infer_fn):
                return self.infer_fn(x, axis_name=batch_axis_name)
            else:
                return self.infer_fn(x)

class FRng(Module):
    """Wrapper to create pure function layers that may require rng."""
    def __init__(
        self,
        train_fn: Callable[[Any, Array], Any],
        infer_fn: Optional[Callable[[Any, Array], Any]]=None
    ):
        """Initialize a FRng layer.

        :param train_fn: Pure function that takes in a valid JAX type, a PRNG
            key, and optionally a keyword argument `axis_name` and returns a
            valid JAX type. Called when ``inference_mode`` is False.
        :param infer_fn: Pure function that takes in a valid JAX type, a PRNG
            key, and optionally a keyword argument `axis_name` and returns a
            valid JAX type. Called when ``inference_mode`` is True. If None, the
            ``train_fn`` is called instead. Default: None.
        """
        super().__init__()
        self.train_fn = train_fn
        self.infer_fn = infer_fn
    
    def init(self, x: Any) -> None:
        pass

    def apply(
        self,
        x: Any,
        rng: Array,
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> Tuple[Any, Any]:
        """Apply an arbitrary pure functional transform."""
        if self.infer_fn is None or not inference_mode:
            if _needs_axis_name(self.train_fn):
                return self.train_fn(x, rng, axis_name=batch_axis_name)
            else:
                return self.train_fn(x, rng)
        else:
            if _needs_axis_name(self.infer_fn):
                return self.infer_fn(x, rng, axis_name=batch_axis_name)
            else:
                return self.infer_fn(x, rng)
