Neural Network Layers
=====================

``mlax.nn`` contains common neural network layers such as ``mlax.nn.Linear`` and
``mlax.nn.Conv``.

``mlax.nn`` also contains meta-layers such as ``mlax.nn.Series`` and
``mlax.nn.Parallel``, which can combine layers in series or parallel.

``mlax.nn`` also contains ``mlax.nn.F`` and ``mlax.nn.FRng``, which are wrappers
that turn pure functions, such as those under ``jax.numpy``, ``jax.nn`` and
``mlax.nn.functional`` into modules.
