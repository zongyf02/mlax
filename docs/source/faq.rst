Frequently Asked Questions
==========================

Layers' ``fwd`` function is throwing an error when jit-compiled.
----------------------------------------------------------------
First ensure that ``jax.jit`` uses the argument ``static_args=["hyperparams"]``.
``hyperparams`` contains Python types such as strings, which not valid JAX
types. ``hyperparams`` can also be used for control-flow. Therefore, they should
be treated as compile-time constants.

If the layers also have different behavior during inference and training, make
sure ``static_args=["hyperparams", "inference_mode"]``. ``inference_mode`` in
such layers is used for control-flow, so should be treated as compile-time
constant.

Finally, ensure that the ``hyperparams`` are hashable. ``jax.jit`` requires
static arguments to be hashable and immutable. Avoid passing lists and other
non-hashable types to the layers' ``init`` function; doing so may result in
non-hashable ``hyperparams``.

How to perform mixed-precision training in mlax?
------------------------------------------------
Most mlax layers' ``init`` function have a ``dtype`` parameter, which controls
the data type of the initialized weights. mlax layers' and blocks' ``fwd``
function will implicitly cast the weights to the dtype of the input features,
meaning the compute type of the forward pass is always the input's dtype.

Therefore, to maintain full-precision weights but compute in half-precision,
simply ensure that each layer receives half-precision inputs.

Some operations/layers, such as softmax and ``mlax.nn.BatchNorm``, need
full-precision for numerical stability. For those layers, cast the input
activations to full-precision, and output activations back to half-precision.

Mixed-precision examples can be found in the Worked Examples.

How to manage device placement in mlax?
---------------------------------------
First, read up on JAX's
`Controlling data and computation placement on devices <https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices>`_.

Since mlax is built with JAX, those apply to mlax model weights. In short, by
default, all variables are on the default devices, usually the first
accelerator. You can use `jax.device_put` to control where you put your
variables and explicitly move data between devices. Also note `numpy` arrays
are always on the CPU and are transferred to the accelerator when used.
