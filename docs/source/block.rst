``mlax.block``
===============

The ``mlax.block`` package contains layers that combine sub-layers. Examples
include ``mlax.block.Series`` and ``mlax.block.Parallel``, which combines the
layers in series and parallel.

Like with those under ``mlax.nn``, modules under ``mlax.block`` each has two
functions:

* ``init`` initializes layer parameters. It returns trainable and non-trainable
    weights, which are two seperate `PyTrees <https://jax.readthedocs.io/en/latest/pytrees.html>`_
    of JAX arrays. It also returns some hyperparameters, which should be a
    NamedTuple of hashable Python types.
* ``fwd`` takes in batched inputs compatible with the weights and
    hyperparameters from ``init``. It figures out the sub-layers from the
    hyperparameters and performs a forward pass on said layers, returning the
    activations and new non-trainable weights.

There are always two versions of the same type of module, for example,
``mlax.block.Series`` and ``mlax.block.Series_rng``.

* A module with ``_rng`` in its name, which only works on layers that do not
    consume a PRNG key.
* A module with ``_rng`` in its name, which consumes a PRNG key, allowing it
    to work on layers requiring PRNG keys as well. However, splitting PRNG keys
    comes at a slight performance cost, so the first option is preferred
    whenever possible.
