API Overview
============

`mlax <https://github.com/zongyf02/mlax>`_ is a pure functional ML library
built on top Google `JAX <https://github.com/google/jax>`_. It is a pure Python
package and is written solely in terms of JAX functions.

In the this overview and docs, any ambiguous term should be interpreted in a
JAX context. For example, ``array`` means ``jax.Array``, and ``dtype`` means JAX
``dtype``.

Unlike most ML libraries, mlax does not support implicit dtype conversions. This
means you cannot, for example, pass ``float16`` inputs to a transformation with
``float32`` weights without getting a runtime error.

This is to avoid the ambiguity that comes with mixed-precision operations. Are
the ``float16`` inputs implicitly converted to ``float32``? or are the
``float32`` weights down-casted to ``float16``? The former follows conventional
type promotion, but the latter is what we expect from mixed-precision in neural
networks.

On the upside, mlax offers strong dtype guarantees. Unless explicitly overriden,
all internal calculations of a function are carried out in the same dtype as its
inputs, and the output will also be in that same dtype.

.. warning::
    mlax does not perform implicit dtype conversions. As a general rule, if a
    function takes in two or more arrays as inputs, they must be of the same
    dtype. All internal operations will be carried out in that dtype and the
    returned values will be of that same dtype.

Since mlax is written in JAX, it is fully compatible with JAX transformations,
notably:

* `grad <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#taking-derivatives-with-grad>`_,
* `vmap <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#auto-vectorization-with-vmap>`_,
* `pmap <https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html>`_, and
* `jit <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#using-jit-to-speed-up-functions>`_

In other words, you get the auto-differentation, auto-vectorization,
parallelization, and jit-compile capabilities of JAX for free. As you will see,
mlax relies on some of thos transformations to properly function.

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
It does not require an implementation seperate from normal convolutions.

.. warning::
    Unless explicitly stated, mlax functions do not operate on batched data nor
    batched weights. Use JAX's ``vmap`` and ``pmap`` transformations to obtain
    batched versions of mlax functions.

mlax contains three subpackages:

.. toctree::
   :maxdepth: 2
   :caption: mlax 

   nn
   blocks
   experimental