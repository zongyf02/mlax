Frequently Asked Questions
==========================

Layers' ``__call__`` function is throwing an error when jit-compiled.
---------------------------------------------------------------------
Ensure that a module's parameters only contain valid JAX types and that its
hyperparameters are comparable and hashable. Jit-compiled functions only operate
on valid JAX types and require their static arguments to be comparable and
hashable.

If the layers also have different behavior during inference and training, make
sure ``static_args="inference_mode"``. ``inference_mode`` in such layers is
used for control-flow, so should be static.

How to perform mixed-precision training in mlax?
------------------------------------------------
Most mlax layers' ``__init__`` function have a ``dtype`` parameter, which
controls the data type of the initialized weights. mlax layers' ``__call__``
function will implicitly cast the weights to the dtype of the input features,
meaning the compute type of the forward pass is always the input's dtype.

Therefore, to maintain full-precision weights but compute in half-precision,
simply ensure that each layer receives half-precision inputs.

Some layers/operations, such as ``mlax.nn.BatchNorm`` and
``mlax.nn.functional.layer_norm``, need full-precision for numerical stability.
For those layers, cast the input activations to full-precision, and output
activations back to half-precision.

Mixed-precision examples can be found on MLAX's GitHub.
