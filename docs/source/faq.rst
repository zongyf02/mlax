Freqently Asked Questions
=========================

Layers' ``fwd`` function is throwing an error when jitted.
-------------------------------------------------------------
First ensure that ``jax.jit`` uses the argument ``static_args=["hyperparams"]``.
``hyperparams`` contains Python types, not valid JAX types. They can also be
used for control-flow. They should be treated as compile-time constant.

If the layers also has different behavior during inference and training, make
sure ``static_args=["hyperparams", "inference_mode"]``. ``inference_mode`` in
such layers is used for control-flow, so should be treated as compile-time
constant.

Finally, ensure that the ``hyperparams`` are hashable. ``jax.jit`` requires
static arguments to be hashable and immutable. Avoid passing lists and other
non-hashable types to the layers' ``init`` function; doing so may result in
non-hashable ``hyperparams``.

How to manage where variables are stored and how data flows in my model?
------------------------------------------------------------------------
First, read up on JAX's
`Controlling data and computation placement on devices <https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices>`_.

Since mlax is built with JAX, those apply to mlax model weights. In short, by
default, all variables are on the default devices, usually the first
accelerator. You can use `jax.device_put` to control where you put your
variables and explictly move data between devices. Also note `numpy` arrays
are always on the CPU and is transferred to the accelerator when used.
