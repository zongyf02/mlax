API Overview
============

`mlax <https://github.com/zongyf02/mlax>`_ is a pure functional ML library
built on top Google `JAX <https://github.com/google/jax>`_. It is a pure Python
package and is written solely in terms of JAX functions.

In the this overview and docs, any ambiguous term should be interpreted in a
JAX context. For example, ``array`` means ``jax.Array``, and ``dtype`` means JAX
``dtype``.

Unlike most ML libraries, mlax do not support implicit type conversions. This
means you cannot, for example, pass ``float16`` inputs to a transformation with
``float32`` weights without getting a runtime error.

This is to avoid ambiguity that comes with mixed-precision operations. Are the
``float16`` inputs implicitly converted to ``float32``? or are the ``float32``
weights down-casted to ``float16``? The former follows conventional type
promotion, but the latter is what we expect from mixed-precision in neural
networks.

.. warning::
    mlax does not implicitly perform type conversions. As a general rule, unless
    otherwise stated, if a function takes in two or more arrays as inputs, they
    must be of the same type.

Since mlax is written in JAX, mlax is fully compatible with JAX transformations,
notably:

* `grad <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#taking-derivatives-with-grad>`_,
* `vmap <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#auto-vectorization-with-vmap>`_,
* `pmap <https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html>`_, and
* `jit <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#using-jit-to-speed-up-functions>`_

In other words, you get the auto-differentation, auto-vectorization,
parallelization, and jit-compile capabilities of JAX for free. As you will see,
mlax relies on some of the above transformations to properly function.

.. note::
    mlax functions are JAX functions; they can be composed with each other
    without issues.

mlax functions are designed to operate on single unbatched samples. Consider
``mlax.nn.linear.bias`` as an exmaple, this function takes in an input ``x`` and 
a bias ``weights``, and returns the element-wise sum of ``x`` plus ``weights``.
This function is only defined when the shape of ``x`` matches that of
``weights``. If ``x`` has an extra batch dimension, the function will result in
a runtime error.

This is to provide maximum flexibility when vectorizing and parallelizing
operations. For example, parallel (grouped) convolutions can be easily achieved
using ``vmap`` in conjunctin with ``pmap`` on the ``mlax.nn.conv2d`` operation.
It does not require a seperate implementation from normal convolutions.

.. warning::
    Unless explicitly stated, mlax functions do not operate on batched data nor
    batched weights. Use JAX's ``vmap`` and ``pmap`` transformations to obtain
    batched versions of mlax functions.

mlax contains three main modules:

* ``nn``
* ``losses``
* ``optim``

``mlax.nn``
-----------
The ``nn`` module contains the building blocks of a neural network. It is
intended to be used with ``jax.nn`` and contains stateful
**atomic transformations** such as ``nn.linear`` and ``nn.bias``, which performs
a linear transformation and adds bias, respectively.

.. note::
    ``jax.nn`` contains statelss operations, ``mlax.nn`` contains stateful
    operations.

Those operations in ``nn`` are called atomic because they cannot be decomposed
into smaller operations (without losing their meaning in a neural network
context).

Atomic transformations have two functions.

* ``init`` takes in hyperparameters and returns some weights for that
    transformation. The weights are usually arrays or tuples of arrays.
* ``fwd`` takes in unbatched inputs, compatible weights from ``init``, and
    performs a forward pass, returning the activations.

.. warning::
    Because mlax does not promote types implicitly, inputs and weights must be
    of the same type. The returned activations will also be of that type.

While atomic operations are great for fine-grained control, they can be
inconvenient if you just want a well-defined neural network layer. Who wants to
compose two functions (``nn.linear`` and ``nn.bias``) just to intialize a dense
layer (linear transformation with bias).

For that reason, the ``nn`` module has a submodule ``nn.blocks``, which contains
common neural network blocks made from the atomic operations.

The ``nn.blocks.Linear`` block, for example, combines the ``linear`` and
``bias`` atomic operations, and also applies an optional activation function.

``nn.blocks`` blocks have two functions, just like atomic operations.

* ``init`` takes in hyperparameters and return block weights. The weights are
    always instances of ``NamedTuple``.
* ``fwd`` takes in unbatched inputs, compatible ``NamedTuple`` weights, and
    returns final activations.

.. warning::
    Just like with atomic operations, ``nn.blocks`` blocks' inputs and weights
    must be of the same type. All internal computations will be carried out in
    that type, and so will be the activations.
    
    If you wish for some internal operations to be done in a different
    type (for example, matrix multiplication in ``float16`` but ``reduce_sum``
    in ``float32``), define your own blocks with atomic operations and explicit
    type casting.

``mlax.losses``
---------------

Coming Soon!

``mlax.optim``
--------------

Coming Soon!
