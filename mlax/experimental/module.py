from jax import (
    tree_util as jtu
)
from dataclasses import dataclass
from typing import Any, Sequence, List

@jtu.register_pytree_node_class
@dataclass(eq=False)
class Parameter:
    """PyTree wrapper around a valid JAX object with metadata marking the object
    as trainable or non-trainable."""
    trainable: bool
    data: Any = None

    def tree_flatten(self):
        """Flatten into a valid JAX object and auxiliary metadata."""
        return (self.data,), (self.trainable,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten a valid JAX object and auxiliary metadata."""
        return cls(*aux, *children)
    
    def copy(self):
        """Use ``tree_flatten`` and ``tree_unflatten`` to make a deep copy."""
        data, treedef = jtu.tree_flatten(self)
        return jtu.tree_unflatten(treedef, data)

def is_parameter(p) -> bool:
    """Whether ``p`` is a Parameter"""
    return isinstance(p, Parameter)

class _ModuleMeta(type):
    """Registers all modules as a PyTree"""
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        jtu.register_pytree_node_class(cls)
        return cls

class Module(metaclass=_ModuleMeta):
    """MLAX layer base class. PyTree of parameters with hyperparameters as
    auxiliary data."""
    def tree_flatten(self):
        """Flatten into parameters and auxiliary hyperparameters."""
        param_names = []
        param_values = []
        hyperparam_names = []
        hyperparam_values = []
        for name, value in vars(self).items():
            if is_parameter(value) or isinstance(value, Module):
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
    
    @property
    def trainables(self) -> List[Parameter]:
        """Return copies of trainable parameters."""
        leaves = jtu.tree_leaves(self, is_leaf=is_parameter)
        return [x.copy() for x in leaves if is_parameter(x) and x.trainable]
            
    @property
    def non_trainables(self) -> List[Parameter]:
        """Return copies of non-trainable parameters."""
        leaves = jtu.tree_leaves(self, is_leaf=is_parameter)
        return [x.copy() for x in leaves if is_parameter(x) and not x.trainable]

    def load(
        self,
        trainables: Sequence[Parameter],
        non_trainables: Sequence[Parameter]
    ):
        """Return a copy of ``self`` with parameters replaced by ``trainables``
        and ``non_trainables`` """
        trainables = iter(trainables)
        non_trainables = iter(non_trainables)
        def _load(leaf):
            if is_parameter(leaf):
                if leaf.trainable:
                    return next(trainables).copy()
                else:
                    return next(non_trainables).copy()
            else:
                return leaf
        
        return jtu.tree_map(
            _load,
            self,
            is_leaf=is_parameter
        )
    
    def __call__(self, x, rng, inference_mode=False):
        raise NotImplementedError()
    
    def fwd(
        self,
        trainables: Sequence[Parameter],
        non_trainables: Sequence[Parameter],
        x: Any,
        rng: Any=None,
        inference_mode: bool=False
    ):
        """Load ``trainables`` and ``non_trainables`` into ``self`` then invoke
        ``__call__``."""
        self = self.load(trainables, non_trainables)
        return self(x, rng, inference_mode) 
