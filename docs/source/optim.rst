``mlax.optim``
==============

The ``mlax.optim`` package contains stateful neural-network optimizers and
stateless helper functions.

Optimizers
----------

Optimizers are stateful modules that update the weights of models based on a
loss function.

In ``mlax``, optimizers have two functions.

* ``init`` takes in optimizer paremeters and model weights, and returns an
    optimizer state that is always a ``NamedTuple``. The values in an optimizer
    state are guaranteed to have the same `pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`_
    structure as the input model weights used to intialize that state. For
    example, ``optim.sgd.init`` returns an ``optim.sgd.State`` which contains
    one value: ``velocity``. Then ``veclocity`` will have have the same pytree
    structure as the ``model_weights`` used to intialize it.
* ``step`` takes in model weight gradients and a compatible optmizer state.
    It returns a pytree of the same strcture as ``model_weights`` and
    ``optimi_state`` that contains the update gradients to the model weights.
    It also returns a new optimizer state.

.. warning::
    mlax does not promote dtypes implicitly. Pytree inputs to the same function
    must have the same structure, and their corresponding leaves must have the
    same dtype. The output pytrees will also have the same structure, with
    the corresponding leaves being the same dtype. This is because, internally,
    mlax uses ``jax.tree_util`` to iterate over the trees leaf by leaf. So
    although dtypes in a pytree can be different, dtypes of respective leaves
    between pytrees must be the same.

If you wish for custom behavior in your optimizers, you can define one yourself
using ``jax.numpy`` and ``jax.tree_util``, or you can take a look in `Optax <https://github.com/deepmind/optax>`_,
which is interoperable with mlax.

Helper functions
-----------------

Helper functions are stateless functions to help apply the update gradients.

Currently, the only helper function is ``mlax.optim.apply_updates``, which
applies update gradients to model weights, returning the new model weights.
Its Haskell signature is ``gradients -> model_weights -> new_model_weights``.
