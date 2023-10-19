"""MLAX module base class and parameter."""
from abc import ABCMeta
from typing import Any, Optional, Tuple, Union, Hashable
from jax import (
    Array,
    tree_util as jtu
)
from mlax._utils import _identity

@jtu.register_pytree_node_class
class Parameter:
    """PyTree wrapper around a valid JAX object and metadata."""

    def __init__(self, trainable: Optional[bool], data: Any=None):
        """Initialize parameter.

        :param trainable: Whether the parameter is trainable or non-trainable.
            If ``None``, indicates that the data field contains nested
            parameters.
        :param data: The content of parameter. Must be a valid JAX type or a
            PyTree of valid JAX types.
            Default: ``None``.
        """
        super().__init__()
        self.trainable = trainable
        self.data = data

    def tree_flatten(self):
        """Flatten into a valid JAX object and auxiliary metadata."""
        meta_names = []
        meta_values = []
        for name, value in vars(self).items():
            if name != "data" and name != "trainable":
                meta_names.append(name)
                meta_values.append(value)

        return (self.data,), (
            self.trainable, tuple(meta_names), tuple(meta_values)
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten a valid JAX object and auxiliary metadata."""
        trainable, meta_names, meta_values = aux
        self = cls(trainable, *children)
        for name, value in zip(meta_names, meta_values):
            object.__setattr__(self, name, value)
        return self

    def __repr__(self) -> str:
        return f"Parameter(trainable={self.trainable}, data={self.data})"

def is_trainable_param(p):
    """Whether ``p`` is a parameter whose ``trainable is True``."""
    return isinstance(p, Parameter) and p.trainable is True

def is_non_trainable_param(p):
    """Whether ``p`` is a parameter whose ``trainable is False``."""
    return isinstance(p, Parameter) and p.trainable is False

def is_leaf_param(p):
    """Whether ``p`` is a parameter whose ``trainable is not None``."""
    return isinstance(p, Parameter) and p.trainable is not None

class _ModuleMeta(ABCMeta):
    """Registers all modules as a PyTree"""
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        jtu.register_pytree_with_keys_class(cls)
        return cls

class Module(metaclass=_ModuleMeta):
    """MLAX layer base class. PyTree of `mlax.Parameters`.
    """
    def __init__(self) -> None:
        """Initialize module hyperparameters."""
        self.initialized = False
        """Whether a module (hyperparameters, parameters, and submodules) is
        initialized."""

    def tree_flatten_with_keys(self):
        """Flatten into parameters and auxiliary hyperparameters."""
        param_names = []
        param_values = []
        hyperparam_names = []
        hyperparam_values = []
        for name, value in vars(self).items():
            if name != "initialized":
                if isinstance(value, (Parameter, Module)):
                    param_names.append(name)
                    param_values.append((name, value))
                else:
                    hyperparam_names.append(name)
                    hyperparam_values.append(value)
        return param_values, (
            param_names, hyperparam_names, hyperparam_values, self.initialized
        )

    @classmethod
    def tree_unflatten(cls, aux, param_values):
        """Unflatten parameters and auxiliary hyperparameters."""
        param_names, hyperparam_names, hyperparam_values, initialized = aux
        self = cls.__new__(cls)
        for name, value in zip(param_names, param_values):
            object.__setattr__(self, name, value)
        for name, value in zip(hyperparam_names, hyperparam_values):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "initialized", initialized)
        return self

    def set_up(self, x: Any) -> None:
        """Initialize parameters. Submodules may not be initialized.

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
        """Perform the forward pass assuming ``set_up`` has been called.
         
         .. note::
            Because ``set_up`` may not initialize submodules, ``forward`` may
            need to initialize submodules before using them. This is commonly
            done recursively by using their ``__call__`` method.

        :param x: Compatible input features.
        :param rng: PRNG key. Only necessary for some modules.

        .. note::
            When overriding, set ``rng``'s default value to ``None`` if a key
            is not required. MLAX uses this information to avoid splitting and
            passing keys to modules that do not need them.

        :param inference_mode: Whether in inference or training mode. Default:
            training mode.
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
        initialized ``self``.

        :param x: Compatible input features.
        :param rng: PRNG key. Only necessary for some modules.
        :param inference_mode: Whether in inference or training mode. Default:
            training mode.
        :param batch_axis_name: Hashable or tuple of hashable representing
            the batch axis name(s) when called in a `jax.vmap` or `jax.pmap`
            context. Used by modules such as `ZNorm` to normalize along the
            batch axis. Default: (), no batch axis.

        :returns: Output features.
        :returns: Initialized ``self``.
        """
        if self.initialized is False:
            self.set_up(x)
        ret = self.forward(x, rng, inference_mode, batch_axis_name)
        self.initialized = True
        return ret, self

    def filter(self, f=is_trainable_param, inverse=False) -> Any:
        """Apply a filter ``f`` on ``self``'s parameters. Filtered out
        parameters have their ``data`` field replaced with ``None``.
        """
        def _filter(arg):
            arg_copy = jtu.tree_map(_identity, arg)
            if (f(arg_copy) ^ inverse):
                return arg_copy
            else:
                arg_copy.data = None
                return arg_copy
        return jtu.tree_map(_filter, self, is_leaf=is_leaf_param)

    def partition(self, f=is_trainable_param) -> Tuple[Any, Any]:
        """Partition on ``self``'s parameters on filter ``f``. Unselected
        parameters have their ``data`` field replaced with ``None``.
        """
        return (self.filter(f, inverse=False), self.filter(f, inverse=True))

    def filter_with_path(self, f, inverse=False) -> Any:
        """``filter`` with path."""
        def _filter_w_path(path, arg):
            arg_copy = jtu.tree_map(_identity, arg)
            if (not f(path, arg_copy) if inverse else f(path, arg_copy)):
                return arg_copy
            else:
                arg_copy.data = None
                return arg_copy
        return jtu.tree_map_with_path(
            _filter_w_path, self, is_leaf=is_leaf_param
        )

    def partition_with_path(self, f) -> Tuple[Any, Any]:
        """``partition`` with path."""
        return (
            self.filter_with_path(f, inverse=False),
            self.filter_with_path(f, inverse=True)
        )

    def combine(self, *rest):
        """Combine ``self``'s parameters with ``rest``'s."""
        def _combine(*args):
            arg_copy = jtu.tree_map(_identity, args[0])
            for arg in args[1:]:
                if isinstance(arg, Parameter) and arg.data is not None:
                    arg_copy.data = arg.data
                    break
            return arg_copy
        return jtu.tree_map(_combine, self, *rest, is_leaf=is_leaf_param)

    def __repr__(self) -> str:
        string = self.__class__.__name__ + "("
        for name, value in vars(self).items():
            string += f"{name}={value}, "
        return string[:-2] + ")"
