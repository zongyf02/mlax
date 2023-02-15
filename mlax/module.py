import jax
from jax import (
    tree_util as jtu
)
from typing import Any, Hashable, Callable

@jtu.register_pytree_node_class
class Parameter:
    """PyTree wrapper around a valid JAX object with metadata."""

    def __init__(self, trainable: bool, data: Any = None, name: Hashable = None) -> None:
        """Initialize parameter."""
        super().__init__()
        self.trainable = trainable
        self.data = data
        self.name = name

    def tree_flatten(self):
        """Flatten into a valid JAX object and auxiliary metadata."""
        return (self.data,), (self.trainable, self.name)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten a valid JAX object and auxiliary metadata."""
        trainable, name = aux
        return cls(trainable, *children, name)

    def __repr__(self) -> str:
        return f"Parameter(trainable={self.trainable}, data={self.data}, name={self.name})"

def is_parameter(p) -> bool:
    """Whether ``p`` is a Parameter."""
    return isinstance(p, Parameter)

def is_trainable(p) -> bool:
    """Whether ``p`` is trainable."""
    return p.trainable

def is_non_trainable(p) -> bool:
    """Whether ``p`` is non_trainable."""
    return p.non_trainable

class _ModuleMeta(type):
    """Registers all modules as a PyTree"""
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        jtu.register_pytree_node_class(cls)
        return cls

class Module(metaclass=_ModuleMeta):
    """MLAX layer base class. PyTree of parameters with hyperparameters as
    auxiliary data.
    """

    def __init__(self)  -> None:
        """Initialize module."""
        super().__init__()

    def tree_flatten(self):
        """Flatten into parameters and auxiliary hyperparameters."""
        param_names = []
        param_values = []
        hyperparam_names = []
        hyperparam_values = []
        for name, value in vars(self).items():
            if is_parameter(value) or is_mlax_module(value):
                param_names.append(name)
                param_values.append(value)
            else:
                hyperparam_names.append(name)
                hyperparam_values.append(value)
        return tuple(param_values), (
            tuple(param_names),
            tuple(hyperparam_names),
            tuple(hyperparam_values)
        )

    @classmethod
    def tree_unflatten(cls, aux, param_values):
        """Unflatten parameters and auxiliary hyperparameters."""
        param_names, hyperparam_names, hyperparam_values = aux
        self = cls.__new__(cls)
        for name, value in zip(param_names, param_values):
            object.__setattr__(self, name, value)
        for name, value in zip(hyperparam_names, hyperparam_values):
            object.__setattr__(self, name, value)
        return self

    def __call__(self, x, rng, inference_mode=False):
        raise NotImplementedError()

    def __repr__(self) -> str:
        repr = self.__class__.__name__ + "("
        for name, value in vars(self).items():
            repr += f"{name}={value}, "
        return repr[:-1] + ")"

    def filter(self, f: Callable[[Parameter], bool], *rest):
        """Apply a filter ``f`` on ``self``'s parameters. Filtered out
        parameters are replaced with a Parameter whose ``trainable = None``.
        """
        return jtu.tree_map(
            _create_true_filter(f), self, *rest, is_leaf=is_parameter
        )

    def partition(self, f: Callable[[Parameter], bool] = is_trainable, *rest):
        """Partition on ``self``'s parameters on filter ``f`` on ``self``'s
        parameters. Unselected parameters are replaced with a Parameter whose
        ``trainable = None``.
        """
        return (
            jtu.tree_map(
                _create_true_filter(f), self, *rest, is_leaf=is_parameter
            ),
            jtu.tree_map(
                _create_false_filter(f), self, *rest, is_leaf=is_parameter
            )
        )

    def combine(self, *rest):
        """Combine ``rest``'s parameters with ``self``'s, with preceding args'
        possibly overriding subsequent args' and ``self``'s.
        """
        def _combine(*args):
            for arg in args:
                if arg.trainable is not None:
                    return arg
            return args[-1]

        return jtu.tree_map(
            _combine, *rest, self, is_leaf=is_parameter
        )

def fwd(
    trainables,
    non_trainables,
    x: Any,
    rng: jax.Array = None,
    inference_mode: bool=False
):
    """Combine ``trainables`` with ``non_trainables`` and invoke ``call``."""
    module = trainables.combine(non_trainables)
    return module(x, rng, inference_mode)

def is_mlax_module(m) -> bool:
    """Whether ``m`` is a module"""
    return isinstance(m, Module)

EMPTY_PARAM = Parameter(trainable=None, data=None, name=None)

def _create_true_filter(f):
    def _filter_true(*args):
        return args[0] if f(*args) else EMPTY_PARAM
    return _filter_true

def _create_false_filter(f):
    def _filter_false(*args):
        return EMPTY_PARAM if f(*args) else args[0]
    return _filter_false
