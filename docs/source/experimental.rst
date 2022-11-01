``mlax.experimental``
======================

The ``mlax.experimental`` package contains modules and functions that are
in testing and/or whose usage is discouraged. They may be moved or removed in
the next release without notice.

Currently there are two subpackages.

* ``mlax.experimental.losses``
* ``mlax.experimental.optim``

``mlax.experimental.losses``
-----------------------------

This module contains loss functions of the Haskell type signature
``predictions -> targets -> loss``.

As with all mlax functions, they take in unbatched predictions and targets and
returns unbatched loss. You can use ``jax.vmap`` to take in batched predictions
and targets, and return per-batch losses.

.. warning::
    ``mlax.experimental.losses`` functions follow their strict mathematical
    definition, which may lead to unexpected results. For example,
    ``mlax.experimental.losses.categorical_crossentropy`` does not clip its
    input predictions, which will result in an ``NaN`` loss if the input
    contains ``0``.

The reason this subpackage is in an experimental state is that
`Optax <https://github.com/deepmind/optax>`_ offers a wider variety of loss
functions, and are just as easy to use.

Optax' loss functions are already vectorized, meaning they work on batched
inputs. Since I cannot think of an occassion where one would not use
``jax.vmap`` on a function in ``mlax._exprimental.losses`` anyways, having
vectorized loss functions may be more convenient.

Althought Optax does not offer the mlax's dtype guarantees, looking at their
source code, it appears that Optax follows ``jax.numpy``'s promotion rules,
which do guarantee that precision is never lost. This is most likely sufficient,
even for mixed-precision training.

``mlax.experimental.optim``
----------------------------

This package contains optimizer modules and a helper function. The helper
function is ``mlax.experimental.optim.apply_gradients``, which applies update
gradients returned from an optimizer to model weights.

Optimizer modules have two functions: ``init`` and ``step``. The former takes in
hyper-paremeters and model weights, and returns an optimizer state. The latter
takes in model weight gradients and the optmizer state, and returns update
gradients and a new optimizer state.

As with ``mlax.experimental.losses``, ``mlax.experimental.optim`` is in an
experimental state because `Optax <https://github.com/deepmind/optax>`_ offers a
wider variety of optimizers.

Optax's optimizers are also pure functions that can be composed with JAX
transformations. It follows the ``jax.numpy`` type promotion rules. And it
outputs optimizer states, which can be partitioned and explicitly managed.

Unless Optax changes one of the above features, its use will be encouraged over
``mlax.experimental.optim``.
