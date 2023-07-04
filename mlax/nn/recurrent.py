from typing import Any, Tuple, Union, Hashable
from jax import (
    Array,
    random,
    lax,
    tree_util as jtu
)
from mlax import Module

def _get_first(xs):
    return jtu.tree_map(lambda xs: xs[0], xs)

def _get_rest(xs):
    return jtu.tree_map(lambda xs: xs[1:], xs)

def _get_first_r(xs):
    return jtu.tree_map(lambda xs: xs[-1], xs)

def _get_rest_r(xs):
    return jtu.tree_map(lambda xs: xs[:-1], xs)

def _expand_dims_concat(x, xs):
    def _f(x, xs):
        return lax.concatenate((lax.expand_dims(x, (0,)), xs), 0)
    return jtu.tree_map(_f, x, xs)

def _expand_dims_concat_r(xs, x):
    def _f(xs, x):
        return lax.concatenate((xs, lax.expand_dims(x, (0,))), 0)
    return jtu.tree_map(_f, xs, x)

class Recurrent(Module):
    """Wrapper around a recurrent cell that does not require rng."""
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
    
    def setup(self, xh: Tuple[Any, Any]) -> None:
        pass

    def forward(
        self,
        xh: Tuple[Any, Any],
        rng: None=None,
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> Any:
        def iteration(acc, x):
            cell, hidden = acc
            (x, hidden), cell = cell(
                (x, hidden), None, inference_mode, batch_axis_name
            )
            return (cell, hidden), x

        xs, hidden = xh
        if self.cell.initialized is False:
            if self.reverse is False:
                (x, hidden), self.cell = self.cell(
                    (_get_first(xs), hidden), None,
                    inference_mode, batch_axis_name
                )
                xs = _get_rest(xs)
                (self.cell, hidden), xs = lax.scan(
                    iteration, (self.cell, hidden), xs,
                    reverse=self.reverse, unroll=self.unroll
                )
                xs = _expand_dims_concat(x, xs)
            else:
                (x, hidden), self.cell = self.cell(
                    (_get_first_r(xs), hidden), None,
                    inference_mode, batch_axis_name
                )
                xs = _get_rest_r(xs)
                (self.cell, hidden), xs = lax.scan(
                    iteration, (self.cell, hidden), xs,
                    reverse=self.reverse, unroll=self.unroll
                )
                xs = _expand_dims_concat_r(xs, x)
        else:
            (self.cell, hidden), xs = lax.scan(
                iteration, (self.cell, hidden), xs,
                reverse=self.reverse, unroll=self.unroll
            )
        return xs, hidden
    
class RecurrentRng(Module):
    """Wrapper around a recurrent cell that may require rng."""
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
    
    def setup(self, xh: Tuple[Any, Any]) -> None:
        pass

    def forward(
        self,
        xh: Tuple[Any, Any],
        rng: Array,
        inference_mode: bool=False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]]=()
    ) -> Any:
        def iteration(acc, x):
            cell, hidden, i = acc
            (x, hidden), cell = cell(
                (x, hidden), random.fold_in(rng, i),
                inference_mode, batch_axis_name
            )
            return (cell, hidden, i + 1), x

        xs, hidden = xh
        if self.cell.initialized is False:
            if self.reverse is False:
                (x, hidden), self.cell = self.cell(
                    (_get_first(xs), hidden),
                    random.fold_in(rng, 0), inference_mode, batch_axis_name
                )
                xs = _get_rest(xs)
                (self.cell, hidden, _), xs = lax.scan(
                    iteration, (self.cell, hidden, 1), xs,
                    reverse=self.reverse, unroll=self.unroll
                )
                xs = _expand_dims_concat(x, xs)
            else:
                (x, hidden), self.cell = self.cell(
                    (_get_first_r(xs), hidden),
                    random.fold_in(rng, 0), inference_mode, batch_axis_name
                )
                xs = _get_rest_r(xs)
                (self.cell, hidden, _), xs = lax.scan(
                    iteration, (self.cell, hidden, 1), xs,
                    reverse=self.reverse, unroll=self.unroll
                )
                xs = _expand_dims_concat_r(xs, x)
        else:
            (self.cell, hidden, _), xs = lax.scan(
                iteration, (self.cell, hidden, 0), xs,
                reverse=self.reverse, unroll=self.unroll
            )
        return xs, hidden
