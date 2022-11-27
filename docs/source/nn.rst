``mlax.nn`` 
===========

The ``mlax.nn`` package contains neural network layers. Examples include
``mlax.nn.Linear`` and ``mlax.nn.Conv``, which applies a linear transformation
and convolutions respectively.

Modules under ``mlax.nn`` each has two functions:

* ``init`` initializes layer parameters. It returns trainable and non-trainable
    weights, which are two seperate `PyTrees <https://jax.readthedocs.io/en/latest/pytrees.html>`_
    of JAX arrays. It also returns some hyperparameters, which should be a
    NamedTuple of hashable Python types.
* ``fwd`` takes in batched inputs compatible with the weights and
    hyperparameters from ``init``, and performs a forward pass, returning the
    activations and new non-trainable weights.

.. warning::
    Because mlax does not promote dtypes implicitly, the inputs and initialized
    weights to ``fwd`` must be of the same type. Unless explicitly overridden,
    the returned activations will also be of that type.

``mlax.nn`` also has two special modules: ``mlax.nn.F`` and ``mlax.nn.F_rng``.
They are used to convert stateless functions, such as those from ``jax.nn`` and
``mlax.functional``, into layers. ``mlax.nn.F`` is for functions that do not
require a PRNG key, ``mlax.nn.F_rng`` is for those that do.

Multiple layers can be combined using ``mlax.blocks``.
