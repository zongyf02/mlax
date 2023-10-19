from typing import Any, Callable, Optional, Tuple, Union, Hashable
from inspect import signature
from jax import Array
from mlax import Module

def _needs_rng(fn):
    return "rng" in signature(fn).parameters.keys()

def _needs_axis_name(fn):
    return "axis_name" in signature(fn).parameters.keys()

class F(Module):
    """Wrapper to create pure functional layers."""
    def __init__(
        self,
        train_fn: Union[Callable[[Any], Any], Callable[[Any], Any]],
        infer_fn: Optional[Union[Callable[[Any], Any], Callable[[Any], Any]]]=None
    ):
        """Initialize a F layer.

        :param train_fn: Pure function that takes in a valid JAX type and
            optionally keyword arguments ``rng`` and ``axis_name``. Returns a
            valid JAX type. Called when ``inference_mode`` is False.
        :param infer_fn: Pure function that takes in a valid JAX type and
            optionally a keyword argument ``rng`` and ``axis_name``. Returns a
            valid JAX type. Called when ``inference_mode`` is True. If None, the
            ``train_fn`` is called instead. Default: None.
        """
        super().__init__()
        self.train_fn = train_fn
        self.infer_fn = infer_fn
    
    def set_up(self, x: Any) -> None:
        pass

    def forward(
        self,
        x: Any,
        rng: Optional[Array],
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> Any:
        kwargs = {}
        if self.infer_fn is None or not inference_mode:
            if _needs_rng(self.train_fn):
                kwargs["rng"] = rng
            if _needs_axis_name(self.train_fn):
                kwargs["axis_name"] = batch_axis_name
            return self.train_fn(x, **kwargs)
        else:
            if _needs_rng(self.infer_fn):
                kwargs["rng"] = rng
            if _needs_axis_name(self.infer_fn):
                kwargs["axis_name"] = batch_axis_name
            return self.infer_fn(x, **kwargs)
