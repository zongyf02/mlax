``mlax.blocks``
===============

The ``mlax.blocks`` package contains functional implementations of common neural
network building layers, such as ``mlax.blocks.Linear`` for a fully connected
layer.

Modules under ``mlax.blocks`` are refered to as **blocks** because are made from
**atomic transformations**.

The ``mlax.blocks.Linear`` block, for example, combines ``mlax.nn.linear``,
``mlax.nn.bias``, and an optionally activation function (such as ``jax.nn.relu``
).

``mlax.blocks`` modules each have two functions, just like those in ``mlax.nn``.

* ``init`` takes in hyperparameters and returns some weights that are always
    instances of ``NamedTuple``.
* ``fwd`` takes in unbatched inputs, compatible weights from ``init``, and
    returns final activations.

.. warning::
    ``mlax.blocks`` modules' inputs and weights must be of the same dtype. All
    internal atomic transformations will be carried out in that dtype, and the
    returned activations will also be of the that dtype.

    If you want a block with custom behavior, such as a fully connected layer
    that multiplies in ``bfloat16`` but accumulates and applies bias in
    ``float32``, build a custom block from atomic transformations.

.. warning::
    Like all mlax functions, ``fwd`` assumes its input to be a single unbatched
    sample. Use ``jax.vmap`` to get a vectorized version of ``fwd`` that
    operates on batched inputs.
