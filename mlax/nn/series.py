from typing import Any, Iterable, Tuple, Union, Hashable, Optional
from jax import (
    Array,
    random
)
from mlax import Module, Parameter

class Series(Module):
    """Combination of layers."""
    def __init__(self, layers: Iterable[Module]):
        """Initialize a Series layer.

        :param layers: Layers to combine in series.
        """
        super().__init__()
        self.layers = Parameter(trainable=None, data=list(layers))

    def set_up(self, x: Any) -> None:
        pass

    def forward(
        self,
        x: Any,
        rng: Optional[Array],
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> Any:
        for i, layer in enumerate(self.layers.data):
            x, self.layers.data[i] = layer(
                x, None if rng is None else random.fold_in(rng, i),
                inference_mode, batch_axis_name
            )
        return x
