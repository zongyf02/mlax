``mlax.functional`` 
===================

The ``mlax.functional`` module contains stateless JAX functions. Examples
include ``mlax.functional.max_pool`` and ``mlax.functional.dropout``, which
apply max-pooling and random dropouts to input features respectively.

They complement the common neural network functions under ``jax.nn``.

They are often used with ``mlax.nn.F`` and ``mlax.nn.F_rng`` to create neural
network layers.
