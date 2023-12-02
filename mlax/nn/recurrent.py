from typing import Any, Tuple, Union, Hashable, Optional
from jax import (
    Array,
    random,
    lax,
    tree_util as jtu
)
from mlax import Module

class Recurrent(Module):
    """Wrapper around a recurrent cell."""
    def __init__(self, cell, reverse: bool=False, unroll: int=1) -> None:
        """Initialize a recurrent layer.

        :param cell: Recurrent cell to scan over a sequence.
        :param reverse: Whether to scan forward or backward (from or to index
            0). Default: False.
        :param unroll: Number of scan iterations to unroll within a single
            iteration of a loop. Default: 1.
        """
        super().__init__()
        self.cell = cell
        self.reverse = bool(reverse)
        self.unroll = int(unroll)
    
    def set_up(self, xh: Tuple[Any, Any]) -> None:
        pass

    def forward(
        self,
        xh: Tuple[Any, Any],
        rng: Optional[Array],
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> Any:
        def _iteration(acc, x):
            cell, hidden, i = acc
            (x, hidden), cell = cell(
                (x, hidden), None if rng is None else random.fold_in(rng, i),
                inference_mode, batch_axis_name
            )
            return (cell, hidden, i + 1), x

        xs, hidden = xh
        if self.cell.is_set_up is False:
            if self.reverse is False:
                x = jtu.tree_map(lambda xs: xs[0], xs)
                xs = jtu.tree_map(lambda xs: xs[1:], xs)
                def _expand_dims_concat(xs, x):
                    return lax.concatenate((lax.expand_dims(x, (0,)), xs), 0)
            else:
                x = jtu.tree_map(lambda xs: xs[-1], xs)
                xs = jtu.tree_map(lambda xs: xs[:-1], xs)
                def _expand_dims_concat(xs, x):
                    return lax.concatenate((xs, lax.expand_dims(x, (0,))), 0)
            (x, hidden), self.cell = self.cell(
                (x, hidden), None if rng is None else random.fold_in(rng, 0),
                inference_mode, batch_axis_name
            )
            (self.cell, hidden, _), xs = lax.scan(
                _iteration, (self.cell, hidden, 1), xs,
                reverse=self.reverse, unroll=self.unroll
            )
            xs = jtu.tree_map(_expand_dims_concat, xs, x)
        else:
            (self.cell, hidden, _), xs = lax.scan(
                _iteration, (self.cell, hidden, 0), xs,
                reverse=self.reverse, unroll=self.unroll
            )
        return xs, hidden
