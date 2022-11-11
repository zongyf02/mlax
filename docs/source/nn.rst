``mlax.nn`` 
===========

The ``mlax.nn`` package contains fundamental operations of a neural network.
Examples include ``mlax.nn.linear`` and ``mlax.nn.dropout``, which applies a
linear transformation and random dropouts, respectively.

.. note::
    ``mlax.nn`` is fully compatible with ``jax.nn`` and can be used together
    without issues.

.. note::
    The modules under ``mlax.nn`` and the functions under ``jax.nn`` are hereby
    referred to as **atomic transformations**. This is because they carry out
    computations that cannot be decomposed into smaller operations (without
    losing their meaning in a neural network context).

Modules under ``mlax.nn`` can be stateful or statelsss. Stateful modules have
two functions, ``init`` to intialize the module state, and ``fwd`` to carry out
the computation. Stateless modules only have ``fwd`` as they have no weights to
initialize.

* ``init`` takes in hyperparameters and returns some weights that are always 
    arrays or tuples of arrays, which are `pytrees <https://jax.readthedocs.io/en/latest/pytrees.html>`_.
* ``fwd`` takes in unbatched inputs, compatible weights from ``init``, and
    performs a forward pass, returning the activations.

.. warning::
    Because mlax does not promote dtypes implicitly, inputs and weights must be
    of the same dtype. The returned activations will also be of that dtype.

.. warning::
    Like all mlax functions, ``fwd`` assumes its input to be a single unbatched
    sample. Use ``jax.vmap`` to get a vectorized version of ``fwd`` that
    operates on batched inputs.

While atomic transformations are great for fine-grained control, it can be
inconvenient if you use them exclusively to build your neural network. For
higher level building blocks, check out ``mlax.blocks``.
