``mlax.experimental``
======================

The ``mlax.experimental`` package contains modules and functions that are in
flux.

Currently there are one subpackage.

* ``mlax.experimental.losses``

``mlax.experimental.losses``
-----------------------------

This module contains loss functions of the Haskell type signature
``predictions -> targets -> loss``.

As with all mlax functions, they take in unbatched predictions and targets and
returns unbatched loss. You can use ``jax.vmap`` to take in batched predictions
and targets, and return per-batch losses.

.. warning::
    ``mlax.experimental.losses`` functions follow their strict mathematical
    definitions, which may lead to unexpected results. For example,
    ``mlax.experimental.losses.categorical_crossentropy`` does not clip its
    input predictions, which will result in an ``NaN`` loss if the input
    contains ``0``.

Currently, an alternative is `Optax <https://github.com/deepmind/optax>`_,
which offers a wider variety of loss functions.

Note that Optax' loss functions are already vectorized, and Optax does not offer
the mlax's dtype guarantees. Looking at their source code, it appears that Optax
follows ``jax.numpy``'s promotion rules, which do guarantee that precision is
never lost.

The reason this subpackage is in an experimental state is that I am not sure yet
whether losses should follow their strict math defitions or have convenience
features such as clipping zeros before calculating crossentropy loss. Maybe I
will divide the operations into atomic operations and blocks.

I am also considering just recommending the use of Optax's loss functions. I do
not see how I can improve on it, other than implementing mlax's dtype guarantee.
