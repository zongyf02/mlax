Modules and Parameters
======================

MLAX modules are PyTrees that inherit from ``mlax.Module``. Fields that are
parameters and submodules are children of the PyTree. All other fields are
considered auxiliary data.

MLAX parameters inherit from ``mlax.Parameter``. They are PyTrees that wrap
some ``data``. ``data`` can be JAX types (``jax.Array``, ``np.array``, etc.),
PyTrees of valid JAX types (list/tuple/dict of ``jax.Array``), or even other
``mlax.Parameter`` s and ``mlax.Module`` s.

Use parameters' ``trainable`` field to indicate whether a parameter is a leaf in
a PyTree. ``trainable=None`` means a parameter contains nested parameters.
``trainable=False`` indicates a leaf parameter whose ``data`` is not trainable.
``trainable=True`` indicates a leaf parameter whose ``data`` is trainable.

MLAX modules' auxiliary data can contain non JAX types (``str``, ``lambda``,
etc.) but they must be comparable and hashable for the module to be
jit-compiled.

The following code illustrate the different possible fields a module can have.

.. code-block:: python

    class Foo(Module):
        def __init__(self):
            self.foo = Parameter(trainable=True, data=None)  # Ok, empty trainable parameter

    class Bar(Module):
        def __init__(self):
            super().__init__()
            self.a = Parameter(trainable=True, data=jnp.ones((3, 4))) # Ok, trainable parameter
            self.b = Parameter(trainable=False, data=np.ones((3, 4))) # Ok, non-trainable parameter
            self.c = Parameter(trainable=False, data=[1, 2, 3]) # Ok, non-trainable PyTree parameter
            # self.d = Parameter(trainable=False, data="abc") # Not ok, not a valid JAX type in parameter
            # self.e = Parameter(trainable=True, data=Foo()) # Not ok, contains a nested parameter/module in a leaf parameter
            self.f = Parameter(trainable=None, data=Foo()) # Ok, `trainable=None` means not a leaf parameter
            self.g = Foo() # Ok, submodule
            self.h = "abc" # Ok, hyperparameter
            self.i = 1 # Ok, hyperparameter
            # self.j = [1, 2, 3] # Not ok, not hashable hyperparameter

.. note::
    MLAX modules use ``vars()`` to determine their fields during PyTree
    flattening and unflattening. This means all variables must be stored in
    ``__dict__``. Avoid storing variables in ``__slots__``.

Custom Modules
--------------

To define a custom module, inherit from ``mlax.Module`` and implement three
functions:

* ``__init__`` to initialize the module's hyperparameters.
* ``set_up`` to initialize the parameters given a sample input. Submodules may not be initialized.
* ``forward`` to perform the forward pass assuming ``set_up`` has been called. May need to initialize submodules; this is usually done automatically with ``__call__``.

MLAX module implements the ``__call__`` function, which ``set_up`` the current
module if not already. The function then calls ``forward`` and returns the
results and an updated ``self``.

.. code-block:: python

    class MyLayer(Module):
        def __init__(self, ...):
            super().__init__()
            ...
        
        def set_up(self, x: Any):
            ...

        def forward(
            self,
            x: Any,
            rng: Optional[Array],
            inference_mode: bool = False,
            batch_axis_name: Union[Hashable, Tuple[Hashable]] = ()
        ):
            ...

Parameters can be mutated in ``forward``. So, it should be easy to implement a
stateful operation like batch norm.

Hyperparameters can also be mutated in ``forward``, but this may result in
recompilation inside a ``jax.jit``.

Model Surgery
-------------

A module can be treated like any other PyTree, but MLAX modules have some
convenience functions to filter, partition, and combine module parameters.

For example, to get a copy of a module's trainable parameters, use
``module.filter(f=is_trainable_param)``.  Non-trainable parameters' ``data`` are
set to ``None``.

To partition a module into trainable and non-trainable parameters, use
``module.partition(f=is_trainable_param)``. This is equivalent to
``module.filter(f=is_trainable_param), module.filter(f=is_non_trainable_param)``

To combine partitioned modules ``trainables`` and ``non_trainables`` into a new
module, use ``trainables.combine(non_trainables)``. Parameters in ``trainables``
are copied and those whose ``data`` is ``None`` will have it replaced with
the ``data`` of the corresponding parameter in ``non_trainables``.
