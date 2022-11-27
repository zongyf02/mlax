API Overview
============

`mlax <https://github.com/zongyf02/mlax>`_ is a pure functional ML library
built on top Google `JAX <https://github.com/google/jax>`_.

mlax functions are JAX functions. They are fully compatible with native JAX
transformations, notably:

* `grad <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#taking-derivatives-with-grad>`_,
* `vmap <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#auto-vectorization-with-vmap>`_,
* `pmap <https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html>`_, and
* `jit <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#using-jit-to-speed-up-functions>`_

.. note::
    mlax is fully compatible with JAX. There are no new concepts to learn if you
    already know JAX.

In mlax, layer weights, hyperparameters, and functions are decoupled. They are
not stored together in an object but in seperate variables.

Functions are imported from mlax. Trainable and non-trainable weights are JAX
arrays stored in distinct PyTrees. Hyperparameters are NamedTuples containing
hashable Python types (usually not valid JAX types).

mlax does not promote types implicitly. This means, for example, passing
`float16` inputs to a layer with `float32` weights will result in a runtime
error. This is to avoid surprises in mixed-precision operations.

mlax consists of 3 sub-packages and modules:

.. toctree::
    :maxdepth: 2
    :caption: mlax

    functional
    nn
    block
