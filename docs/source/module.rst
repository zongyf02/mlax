State Management
================

All MLAX states inherit from ``mlax.State``.

A state can either by a leaf/exterior node (``mlax.ENode``) or a branch/interior
node ``mlax.INode``.

There are four stateful classes in MLAX.

* ``mlax.ENode``

    * ``mlax.Variable``
    * ``mlax.Parameter``

* ``mlax.INode``

    * ``mlax.Container``
    * ``mlax.Module``

MLAX variables are PyTrees that wrap some non-trainable ``data``. ``data`` can
be JAX types (``jax.Array``, ``np.array``, etc.) or PyTrees
(list/tuple/dict of ``jax.Array``) of valid JAX types. ``data`` should not
contain a nested ``mlax.State``.

MLAX parameters are PyTrees that wrap some trainable ``data``. ``data`` can be
JAX types or PyTrees of valid JAX types.

* Use parameters' ``frozen`` field to indicate whether a parameter should be optimized.

MLAX containers are PyTrees that wrap some nested ``states``. ``states`` must be
MLAX states or PyTrees of MLAX states.

MLAX modules are PyTrees whose variables, parameters, containers, and submodules
are children of the PyTree. All other fields are considered auxiliary data.

MLAX modules' auxiliary data can contain non JAX types (``str``, ``lambda``,
etc.) but they must be comparable and hashable for the module to be
jit-compiled.

The following code illustrate the possible fields a module.

.. code-block:: python

    class Foo(Module):
        pass

    class Bar(Module):
        def __init__(self):
            super().__init__()
            self.a = Parameter(data=jnp.ones((3, 4))) # Ok, unfrozen parameter
            self.b = Parameter(data=np.ones([1, 2, 3]), frozen=True) # Ok, frozen parameter
            # self.c = Parameter(data="abc") # Not ok, not a valid JAX type
            # self.d = Parameter(data=Foo()) # Not ok, contains a nested state
            self.e = Variable(data=jnp.ones((3, 4))) # Ok, Variable
            # self.f = Variable(data="abc") # Not ok, not a valid JAX type
            # self.g = Variable(data=Foo()) # Not ok, contains a nested state
            self.h = Container(states=[Foo(), Foo()]) # Ok, state
            self.i = Container(states="abc") # Not ok, not a valid State
            self.j = Foo() # Ok, submodule
            self.k = "abc" # Ok, hyperparameter
            self.l = 1 # Ok, hyperparameter
            # self.m = [1, 2, 3] # Not ok, not hashable hyperparameter

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

A INode can be treated like any other PyTree, but INode have some functions to
faciliate maniplate the ENodes.

* ``module.filter``
* ``model.filter_with_path``
* ``model.partition``
* ``model.partition_with_path``
* ``model.combine``
