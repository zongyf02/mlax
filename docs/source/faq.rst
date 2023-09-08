Frequently Asked Questions
==========================

Modules' ``__call__`` function is throwing an error when jit-compiled.
----------------------------------------------------------------------
Ensure that a module's parameters only contain valid JAX types and that its
hyperparameters are comparable and hashable. Jit-compiled functions only operate
on valid JAX types and require their static arguments to be comparable and
hashable.

Modules' ``filter`` / ``partition`` functions are throwing ``AttributeError: cannot filter/partition an uninitialized module``
--------------------------------------------------------------------------------------------------------------------------------
Uninitialized modules have unset parameters and/or uninitialized submodules. So,
filtering and partitioning their parameters are disallowed. Initialize the
modules by using the ``__call__`` function on sample inputs.

How to perform mixed-precision training in mlax?
------------------------------------------------
Most MLAX layers' ``__init__`` function have a ``dtype`` parameter, which
controls the data type of the initialized weights. MLAX layers' ``__call__``
function will implicitly cast the weights to the dtype of the input features,
meaning the compute type of the forward pass is always the input's dtype.

Therefore, to maintain full-precision weights but compute in half-precision,
simply ensure that each layer receives half-precision inputs.

Some layers/operations, such as ``mlax.nn.ZNorm`` and ``mlax.nn.functional.z_norm``,
require full-precision for numerical stability. For those layers, cast the input
activations to full-precision, and output activations back to half-precision.
