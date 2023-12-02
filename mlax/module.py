"""MLAX module base class, Container, Variable, and Parameter."""
from abc import ABCMeta
from typing import Any, Callable, Optional, Tuple, Union, Hashable
from jax import (
    Array,
    tree_util as jtu
)
from mlax._utils import _identity

class _PytreeMeta(ABCMeta):
    """Registers all classes and derived classes as a PyTree."""
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        jtu.register_pytree_with_keys_class(cls)
        return cls
    
class State(metaclass=_PytreeMeta):
    """Stateful classes in MLAX."""
    def tree_flatten_with_keys(self):
        """Flatten into a valid JAX type and auxiliary metadata."""
        raise NotImplementedError()
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten a valid JAX type and auxiliary metadata."""
        raise NotImplementedError()

def is_state(arg):
    """Whether ``arg`` is State."""
    return isinstance(arg, State)

class ENode(State):
    """MLAX Pytree leaf/exterior node."""

    def __init__(self, data: Any=None):
        """Initialize ENode

        :param data: The content of ENode. Must be a valid JAX type or a PyTree
            of valid JAX types. Default: ``None``.
        """
        super().__init__()
        self.data = data

    def tree_flatten_with_keys(self):
        meta_names = []
        meta_values = []
        for name, value in vars(self).items():
            if name != "data":
                meta_names.append(name)
                meta_values.append(value)

        return (("data", self.data),), (tuple(meta_names), tuple(meta_values))

    @classmethod
    def tree_unflatten(cls, aux, children):
        meta_names, meta_values = aux
        self = cls(*children)
        for name, value in zip(meta_names, meta_values):
            object.__setattr__(self, name, value)
        return self

def is_leaf_state(arg):
    """Whether ``arg`` is a ENode."""
    return isinstance(arg, ENode)

class Variable(ENode):
    """PyTree wrapper around a non-trainable JAX type."""

    def __init__(self, data: Any=None):
        """Initialize Variable.

        :param data: The content of Variable. Must be a valid JAX type or a
            PyTree of valid JAX types. Default: ``None``.
        """
        super().__init__(data=data)

    def __repr__(self) -> str:
        return f"Variable(data={self.data})"

def is_variable(arg: Any):
    """Whether ``arg`` is a Variable."""
    return isinstance(arg, Variable)

class Parameter(ENode):
    """PyTree wrapper around a trainable JAX type that can be frozen or not."""

    def __init__(self, data: Any=None, frozen: bool=False):
        """Initialize Parameter.
        
        :param data: The content of Parameter. Must be a valid JAX type or a
            PyTree of valid JAX types. Default: ``None``.
        :param frozen: Whether the Parameter is frozen. Default: False.
        """
        super().__init__(data=data)
        self.frozen = frozen

    def __repr__(self) -> str:
        return f"Parameter(data={self.data}, frozen={self.frozen})"

def is_parameter(arg: Any):
    """Whether ``arg`` is a Parameter."""
    return isinstance(arg, Parameter)

def is_frozen_param(arg: Any):
    """Whether ``arg`` is a Parameter and ``frozen`` is True."""
    return is_parameter(arg) and arg.frozen

def is_unfrozen_param(arg: Any):
    """Whether ``arg`` is a Parameter and ``frozen`` is False."""
    return is_parameter(arg) and not arg.frozen

class INode(State):
    """MLAX Pytree branch/inner node."""

    def filter(
        self, f: Callable[[State], bool]=is_unfrozen_param, inverse=False
    ) -> Any:
        """Filter leaf States.

        :param f: Filter to apply on leaf States. Default: is_unfrozen_param.
        :param inverse: Whether to apply the inverse filter. Default: False.

        :returns: New ``self`` whose filtered out leaf States have their
            ``data`` field replaced with ``None``.
        """
        def _filter(arg):
            arg_copy = jtu.tree_map(_identity, arg)
            if not is_leaf_state(arg_copy) or f(arg_copy) ^ inverse:
                return arg_copy
            else:
                arg_copy.data = None
                return arg_copy
        return jtu.tree_map(_filter, self, is_leaf=is_leaf_state)

    def partition(
        self, f: Callable[[ENode], bool]=is_unfrozen_param
    ) -> Tuple[Any, Any]:
        """Partition leaf States.

        :param f: Select function to apply on leaf States (ENodes). Default:
            is_unfrozen_param.

        :returns: New ``self`` whose selected leaf States have their ``data``
            field replaced with ``None``.
        :returns: New ``self`` whose not selected leaf States have their
            ``data`` field replaced with ``None``.
        """
        return (self.filter(f, inverse=False), self.filter(f, inverse=True))

    def filter_with_path(self, f=Callable[[Any, ENode], bool], inverse=False) -> Any:
        """Filter leaf States with their path.

        :param f: Filter to apply on path and leaf States (ENodes).
        :param inverse: Whether to apply the inverse filter. Default: False.

        :returns: New ``self`` whose filtered out leaf States have
            their ``data`` field replaced with ``None``.
        """
        def _filter(path, arg):
            arg_copy = jtu.tree_map(_identity, arg)
            if not is_leaf_state(arg_copy) or f(arg_copy) ^ inverse:
                return arg_copy
            else:
                arg_copy.data = None
                return arg_copy
        return jtu.tree_map_with_path(_filter, self, is_leaf=is_leaf_state)

    def partition_with_path(self, f: Callable[[Any, ENode], bool]) -> Tuple[Any, Any]:
        """Partition leaf States with their path.

        :param f: Select function to apply on path and leaf States.

        :returns: New ``self`` whose selected leaf States have their ``data``
            field replaced with ``None``.
        :returns: New ``self`` whose not selected leaf States have their
            ``data`` field replaced with ``None``.
        """
        return (
            self.filter_with_path(f, inverse=False),
            self.filter_with_path(f, inverse=True)
        )

    def combine(self, *rest) -> Any:
        """Combine ``self``'s Parameters with ``rest``'s.
        
        :param rest: Rest of modules to copy leaf States from.

        :returns: New ``self`` whose empty leaf States (those whose ``data``
            field is ``None``) are replaced the first corresponding non-empty
            leaf States from ``rest``.
        """
        def _combine(*args):
            combined_arg = args[0]
            if combined_arg.data is not None:
                return jtu.tree_map(_identity, combined_arg)
            
            for arg in args[1:]:
                if arg.data is not None:
                    combined_arg = arg
                    break
            return jtu.tree_map(_identity, combined_arg)

        return jtu.tree_map(_combine, self, *rest, is_leaf=is_leaf_state)

def is_branch_state(arg):
    """Whether ``arg`` is a INode."""
    return isinstance(arg, INode)

class Container(INode):
    """MLAX container. PyTree of MLAX States."""
    def __init__(self, states: Any) -> None:
        """Initialize Container.

        :param states: The content of Container. Must be a MLAX State or a
            PyTree of MLAX States. Default: ``None``.
        """
        super().__init__()
        self.states = states

    def tree_flatten_with_keys(self):
        meta_names = []
        meta_values = []
        for name, value in vars(self).items():
            if name != "states":
                meta_names.append(name)
                meta_values.append(value)

        return (("states", self.states),), (tuple(meta_names), tuple(meta_values))

    @classmethod
    def tree_unflatten(cls, aux, children):
        meta_names, meta_values = aux
        self = cls(*children)
        for name, value in zip(meta_names, meta_values):
            object.__setattr__(self, name, value)
        return self

def is_container(arg):
    """Whether ``arg`` is a Container."""
    return isinstance(arg, Container)

class Module(INode):
    """MLAX layer base class. PyTree of Parameters and Variables."""
    def __init__(self) -> None:
        """Initialize module hyperparameters and Variables."""
        super().__init__()
        self.is_set_up = False
        """Whether a module has been set up."""

    def set_up(self, x: Any) -> None:
        """Initialize Parameters and Variables. Submodules may not be set up.

        :param x: Compatible input features.
        """
        raise NotImplementedError()

    def forward(
        self,
        x: Any,
        rng: Optional[Array],
        inference_mode: bool = False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]] = ()
    ) -> Any:
        """Perform the forward pass assuming module is set up.

         .. note::
            Because ``set_up`` may not set up submodules, ``forward`` may
            need to initialize submodules before using them. This is commonly
            done recursively by using their ``__call__`` method.

        :param x: Compatible input features.
        :param rng: PRNG key. Only necessary for some modules.

        .. note::
            When overriding, set ``rng``'s default value to ``None`` if a key
            is not required. MLAX uses this information to avoid splitting and
            passing keys to modules that do not need them.

        :param inference_mode: Whether in inference or training mode. Default:
            False, training mode.
        :param batch_axis_name: Hashable or tuple of hashable representing
            the batch axis name(s) when called in a ``jax.vmap`` or ``jax.pmap``
            context. Used by modules such as ``ZNorm`` to normalize along the
            batch axis. Default: (), no batch axis.

        :returns: Output features.
        """
        raise NotImplementedError()

    def __call__(
        self,
        x: Any,
        rng: Optional[Array],
        inference_mode: bool = False,
        batch_axis_name: Union[Hashable, Tuple[Hashable]] = ()
    ) -> Tuple[Any, Any]:
        """``set_up`` if needed then ``forward``, returning the results and
        fully initialized ``self`` (initialized hyperparameters, Variables,
        Parameters, and set up submodules).

        :param x: Compatible input features.
        :param rng: PRNG key. Only necessary for some modules.
        :param inference_mode: Whether in inference or training mode. Default:
            training mode.
        :param batch_axis_name: Hashable or tuple of hashable representing
            the batch axis name(s) when called in a `jax.vmap` or `jax.pmap`
            context. Used by modules such as `ZNorm` to normalize along the
            batch axis. Default: (), no batch axis.

        :returns: Output features.
        :returns: Fully initialized ``self``.
        """
        if self.is_set_up is False:
            self.set_up(x)
        ret = self.forward(x, rng, inference_mode, batch_axis_name)
        self.is_set_up = True
        return ret, self

    def tree_flatten_with_keys(self):
        param_names = []
        param_values = []
        hyperparam_names = []
        hyperparam_values = []
        for name, value in vars(self).items():
            if isinstance(value, State):
                param_names.append(name)
                param_values.append((name, value))
            else:
                hyperparam_names.append(name)
                hyperparam_values.append(value)
        return param_values, ( param_names, hyperparam_names, hyperparam_values)

    @classmethod
    def tree_unflatten(cls, aux, param_values):
        param_names, hyperparam_names, hyperparam_values = aux
        self = cls.__new__(cls)
        for name, value in zip(param_names, param_values):
            object.__setattr__(self, name, value)
        for name, value in zip(hyperparam_names, hyperparam_values):
            object.__setattr__(self, name, value)
        return self

    def __repr__(self) -> str:
        string = self.__class__.__name__ + "("
        for name, value in vars(self).items():
            string += f"{name}={value}, "
        return string[:-2] + ")"

def is_module(arg: Any):
    """Whether ``arg`` is a Module."""
    return isinstance(arg, Module)
