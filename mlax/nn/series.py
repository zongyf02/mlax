from jax import (
    Array,
    random
)
from typing import Any, Iterable, Tuple, Union, Hashable
from mlax import Module, Parameter
from mlax._utils import _needs_rng

class Series(Module):
    """Combination of layers that do not require rng in series."""
    def __init__(self, layers: Iterable[Module]):
        """Initialize a Series layer.

        :param layers: Layers to combine in series.
        """
        super().__init__()
        self.layers = Parameter(trainable=None, data=list(layers))

    def init(self, x: Any) -> None:
        pass

    def apply(
        self,
        x: Any,
        rng: None=None,
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> Tuple[Any, Any]:
        for i, layer in enumerate(self.layers.data):
            x, self.layers.data[i] = layer(
                x, None, inference_mode, batch_axis_name
            )
        return x

class SeriesRng(Module):
    """Combination of layers that may require rng in series."""
    def __init__(self, layers: Iterable[Module]):
        """Initialize a SeriesRNG layer.

        :param layers: Layers to combine in series.
        """
        super().__init__()
        self.layers = Parameter(trainable=None, data=list(layers))

    def init(self, x: Any) -> None:
        pass

    def apply(
        self,
        x: Any,
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

        for i, (needs_rng, layer) in enumerate(
            zip(needs_rngs, self.layers.data)
        ):
            if needs_rng:
                x, self.layers.data[i] = layer(
                    x, next(keys_iter), inference_mode, batch_axis_name
                )
            else:
                x, self.layers.data[i] = layer(
                    x, None, inference_mode, batch_axis_name
                )
        return x
