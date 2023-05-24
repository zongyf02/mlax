from jax import (
    Array,
    random
)
from typing import Any, Iterable, Tuple, List, Union, Hashable
from mlax import Module, Parameter
from mlax._utils import _needs_rng

class Parallel(Module):
    """Combination of layers that do not require rng in parallel."""
    def __init__(self, layers: Iterable[Module]):
        """Initialize a Parallel layer.

        :param layers: Layers to combine in parallel.
        """
        super().__init__()
        self.layers = Parameter(trainable=None, data=list(layers))

    def init(self, x: Any) -> None:
        pass

    def apply(
        self,
        x: Iterable[Any],
        rng: None=None,
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> Tuple[List[Any], Any]:
        res = []
        for i, (layer, _x) in enumerate(zip(self.layers.data, x)):
            _y, self.layers.data[i] = layer(
                _x, None, inference_mode, batch_axis_name
            )
            res.append(_y)
        return res

class ParallelRng(Module):
    """Combination of layers that may require rng in parallel."""
    def __init__(self, layers: Iterable[Module]):
        """Initialize a ParallelRng layer.

        :param layers: PyTree of layers to combine in parallel.
        """
        super().__init__()
        self.layers = Parameter(trainable=None, data=list(layers))

    def init(self, x: Any) -> None:
        pass

    def apply(
        self,
        x: Iterable[Any],
        rng: Array,
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> Tuple[Any, Any]:
        needs_rngs = [_needs_rng(layer) for layer in self.layers.data]
        n_needs_rng = sum(needs_rngs)
        if n_needs_rng > 1:
            keys_iter = iter(
                [random.fold_in(rng, i) for i in range(n_needs_rng)]
            )
        else:
            keys_iter = iter([rng])

        res = []
        for i, (needs_rng, layer, _x) in enumerate(
            zip(needs_rngs, self.layers.data, x)
        ):
            if needs_rng:
                _x, self.layers.data[i] = layer(
                    _x, next(keys_iter), inference_mode, batch_axis_name
                )
            else:
                _x, self.layers.data[i] = layer(
                    _x, None, inference_mode, batch_axis_name
                )
            res.append(_x)
        return res
