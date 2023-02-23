Modules and Parameters
======================

MLAX modules inherit from ``mlax.Module``. They are PyTrees whose children
are parameters and other modules. Everything else is auxiliary data.

MLAX parameters inherit from ``mlax.Parameter``. They are PyTrees that wrap
some ``data``. ``trainable`` and ``name`` are auxiliary metadata.

Parameters must wrap valid JAX types (``jax.Array``, ``np.array``, etc.) or
PyTrees of valid JAX types if they were to be used in a jit-compiled module.

.. note::
    This is because jit-compiled JAX functions only operate on JAX types or
    PyTrees of JAX types.

However, they should not wrap submodules. Otherwise, ``mlax.Module``'s ``map``,
``filter``, and ``partition`` functions may not work properly.

.. note::
    This is because those functions treat parameters as leaf nodes of the module
    PyTree. Thus they will not recursively flatten any submodule wrapped by a
    parameter.

If your module contains submodules, store them directly as an attribute or wrap
them in a module container such as ``mlax.ModuleSeq``, which inherits from
``mlax.Module`` and stores a mutable sequence of submodules. 

Non JAX types (``str``, ``lambda``, etc.) are treated as hyperparameters and
are stored in auxiliary data. They must be comparable and hashable if they were
to be used in a jit-compiled module.

.. note::
    This is because jit-compiled JAX functions' static arguments must be
    comparable and hashable, and hyperparameters are treated as static
    arguments.

.. warning::
    MLAX modules use ``vars()`` to determine their fields during flattening and
    unflattening. This means all variables must be stored in ``__dict__``.
    Avoid storing variables in ``__slots__``.

.. code-block:: python

    class Foo(Module):
        pass

    class Bar(Module):
        def __init__(self):
            super().__init__()
            self.a = Parameter(trainable=True, data=jnp.ones((3, 4))) # Ok, trainable parameter
            self.b = Parameter(trainable=False, data=np.ones((3, 4))) # Ok, non-trainable parameter
            self.c = Parameter(trainable=False, data=[1, 2, 3]) # Ok, non-trainable parameter PyTree
            # self.d = Parameter(trainable=False, data="abc") # Not ok, not a valid JAX type
            # self.e = Parameter(trainable=False, data=Foo()) # Not ok, should not wrap submodules
            self.f = Foo() # Ok, module can contain submodules
            self.g = ModuleSeq(submodules=[Foo()]) # Ok, module can contain module containers, which are submodules
            self.h = "abc" # Ok, hyperparameter
            self.i = 1 # Ok, hyperparameter
            # self.j = [1, 2, 3] # Not ok, not hashable

To define a custom layer, inherit from ``mlax.Module`` and implement a forward
pass function with ``__call__(self, x, rng, inference_mode=False)`` or
``__call__(self, x, rng=None, inference_mode=False)``.

``__call__`` must return a pair whose first element is the result of the
computation on ``x``, and whose second element is ``self``.

Use the former signature if ``__call__`` requires a PRNGKey. Use the latter if
it does not.

.. code-block:: python

    class MyLayer(Module):
        def __init__(self, ...):
            super().__init__()
            ...
        
        def __call__(self, x, rng, inference_mode=False):
            ...
            return x, self

In ``__call__``, parameters can be directly mutated, provided that the updated
``self`` is returned in the end.

.. warning::
    Ensure a module's ``__call__`` returns not only the result on the forward
    pass but also the updated ``self``.

Hyperparameters can be mutated, but they must be assigned static values if they
were to be used inside ``jax.jit``.

.. warning::
    Jit-compiled JAX functions are retraced when their static arguments change.
    Since hyperparameters are treated as static arguments, if a hyperparameter
    can take on more than a few values, consider making it a parameter to avoid
    excessive retracing.

.. code-block:: python

        @partial(jax.jit, static_argnames="inference_mode")
        def __call__(self, x, rng=None, inference_mode=False):
            self.a.data = jnp.zeros((3, 4)) # Ok, update trainable parameter
            self.e = "bcd" # Ok, update hyperparameter to static value
            self.e = inference_mode # Ok, update hyperparameter to static value
            # self.e = x # Not ok, updating hyperparameter to a traced value.
            self.f += 1 # Allowed, but all subsequent calls will be retraced. Better to make self.f a Parameter.

A module can be treated like any other PyTree, but ``mlax.Module`` has some
convenience functions to help map, filter, partition, and combine module
parameters.

To map a function over all parameters, use ``new_module = module.map(f)``.

.. note::
    Compared to ``jax.tree_util.tree_map``, ``mlax.Module``'s ``map`` treats
    ``mlax.Parameters`` as leaves. In other words, it will not flatten
    parameter PyTrees, which is desirable if one wishes to access their metadata
    (``trainable`` and ``name``).

To filter a module's parameters, for example, to get only the trainable
parameters, use ``trainables = module.filter(f=is_trainable)``. ``trainables``
is identical to ``module`` except that non-trainable parameters are replaced
with a special value: ``mlax.Parameter(trainable=None, data=None)``.

To partition a module, for example, into trainable and non-trainable parameters,
use ``trainables, non_trainables = module.partition(f=is_trainable)``.
``trainables`` and ``non_trainables`` are identical to ``module`` except that
non-trainable parameters in ``trainables`` and trainable parameters in
``non_trainables`` are replaced with the special value
``mlax.Parameter(trainable=None, data=None)``.

Use ``mlax.Module.combine`` to combine partitioned modules:
``module = trainables.combine(non_trainables)``. Each parameter in ``module``
is equal to the first non special-valued parameter (one whose ``trainable`` is
not None) from ``trainables`` and ``non_trainables``.
