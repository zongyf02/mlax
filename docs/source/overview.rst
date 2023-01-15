Overview
========

`MLAX <https://github.com/zongyf02/mlax>`_ is a pure functional ML library
built on top Google `JAX <https://github.com/google/jax>`_.

MLAX follows object-oriented semantics like Keras and PyTorch.

Modules are PyTrees whose leaves are parameters and whose auxiliary data
(metadata) are hyperparameters.

This means MLAX is fully compatible with native JAX transformations, notably:

* `grad <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#taking-derivatives-with-grad>`_,
* `vmap <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#auto-vectorization-with-vmap>`_,
* `pmap <https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html>`_, and
* `jit <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#using-jit-to-speed-up-functions>`_

Why MLAX?
---------

Compared to existing libraries such as Flax and Haiku,

* MLAX stores parameters in the modules themselves rather than in a seperate
    structure, making layer development and parameter surgery easier.
* MLAX does not require sepcial versions ``grad``, ``vmap``, ``pmap``, and
    ``jit``, making it easier to learn and integrate with other JAX libraries.

Compared to other PyTree module-based libraries such as Equinox and Treex,

* MLAX employs an explicit state managment system with minimal magic, keeping
    the code purely functional. If you mutated something, you must return it!
* MLAX allows parameters and hyperparameters to be updated in the forward pass,
    making it easy to develop layers like BatchNorm.
* MLAX never partitions or returns partial modules, guaranteeing all modules
    to be callable.

Installation
-------------

Visit MLAX's `GitHub page <https://github.com/zongyf02/mlax#installation>`_.

Worked Examples
---------------

End-to-end examples with reference PyTorch implementations can be found on
`GitHub <https://github.com/zongyf02/mlax/tree/main/examples>`_ as well.
