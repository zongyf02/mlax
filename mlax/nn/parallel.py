from typing import Any, Iterable, Tuple, List, Union, Hashable, Optional
from jax import (
    Array,
    random
)
from mlax import Container, Module

class Parallel(Module):
    """Combination of layers in parallel."""
    def __init__(self, layers: Iterable[Module]):
        """Initialize a Parallel layer.

        :param layers: Layers to combine in parallel.
        """
        super().__init__()
        self.layers = Container(list(layers))

    def set_up(self, x: Any) -> None:
        pass

    def forward(
        self,
        x: Iterable[Any],
        rng: Optional[Array],
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> List[Any]:
        res = []
        for i, (layer, _x) in enumerate(zip(self.layers.states, x)):
            _y, self.layers.states[i] = layer(
                _x, None if rng is None else random.fold_in(rng, i),
                inference_mode, batch_axis_name
            )
            res.append(_y)
        return res
