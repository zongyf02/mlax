Modules and Parameters
======================

MLAX modules inherit from ``mlax.Module``. They are PyTree nodes whose children
are parameters and other modules. Everything else is auxiliary data.

MLAX parameters inherit from ``mlax.Parameter``. They are also PyTree nodes
whose children is ``parameter.data``. ``parmeter.trainable`` is auxiliary data.

Parameters must contain valid JAX types (``jax.Array``, ``np.array``, etc.) or
PyTrees of valid JAX types if they were to be used in a jit-compiled module.

.. note::
    This is because jit-compiled JAX functions only operate on JAX types or
    PyTrees of JAX types.

Non JAX types (``str``, ``lambda``, etc.) will be treated as hyperparameters and
be stored in auxiliary data. They must be comparable and hashable if they were
to be used in a jit-compiled module.

.. note::
    This is because jit-compiled JAX functions' static arguments must be
    comparable and hashable.

.. code-block:: python

    class Foo(Module):
        def __init__(self):
            super().__init__()
            self.a = Parameter(trainable=True, data=jnp.ones((3, 4))) # Ok, trainable parameter
            self.b = Parameter(trainable=False, data=np.ones((3, 4))) # Ok, non-trainable parameter
            self.c = Parameter(trainable=False, data=[1, 2, 3]) # Ok, non-trainable parameter PyTree
            # self.d = Parameter(trainable=False, data="abc") # Not ok, not a valid JAX type
            self.e = "abc" # Ok, hyperparameter
            self.f = 1 # Ok, hyperparameter
            # self.g = [1, 2, 3] # Not ok, not hashable

        ...
    
In the forward call, parameters can be directly mutated, provided that ``self``
is returned in the end.

Hyperparameters can be mutated, but they must be assigned static values if they
were to be used inside ``jax.jit``.

.. warning::
    Modules with different parameters will be retraced when called. If a
    hyperparameter can take on more than a few values, consider making it a
    parameter to avoid excessive retracing.

.. code-block:: python

        @partial(jax.jit, static_argnames="inference_mode")
        def __call__(self, x, rng=None, inference_mode=False):
            self.a.data = jnp.zeros((3, 4)) # Ok, update trainable parameter
            self.e = "bcd" # Ok, update hyperparameter to static value
            self.e = inference_mode # Ok, update hyperparameter to static value
            # self.e = x # Not ok, updating hyperparameter to a traced value.
            self.f += 1 # Allowed, but all subsequent calls will be retraced. Better to make self.f a Parameter.

To get a snapshot of a module's trainable parameters, use
``trainables = module.trainables``. ``traiables`` will be assigned a list of all
Parameters whose ``trainable=True``. Later mutations of ``module`` will not be
reflected in ``trainables``.

Similarly, to get a snapshot of a module's non-trainable parameters, use
``non_trainables = module.non_trainables``. ``non_traiables`` will be assigned a
list of all Parameters whose ``trainable=False``. Later mutations of ``module``
will not be reflected in ``non_trainables``.

Use ``mlax.Module.load_trainables`` and ``mlax.Module.load_non_trainables`` to
return a new instance of the module whose trainables and non-traiables are
overwritten.

To define a custom layer, inherit from ``mlax.Module`` and implement the
``__call__(self, x, rng, inference_mode=False)`` or
``__call__(self, x, rng=None, inference_mode=False)``.

``__call__`` must return a pair whose first element is the result on the
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
