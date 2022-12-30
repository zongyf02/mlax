Dev Notes
==========

Planned features ranked by priority
------------------------------------
Attention, recurrent layers.

Is there a performance penalty for having unused function parameters?
----------------------------------------------------------------------
In eager mode, yes, albeit a small one.

In a jit-compiled function, JAX is smart enough to drop unused arguments from
XLA executables. ``jax.jit`` has a ``keep_unused`` parameter to control that.
You can also verify this by
`inspecting the lowered function <https://jax.readthedocs.io/en/latest/aot.html#inspecting-staged-out-computations>`_.

However, if the unused parameters cannot be jit-compiled, and we list them under
``jax.jit``'s ``static_argnames``, then the function will be retraced every time
new arguments are passed into the function.

In the case of the ``inference_mode`` parameter that every ``fwd`` function in
``mlax.nn`` has, if it is not used by the forward function, then do not include
it in ``static_argnames`` when jit-compiling to prevent unnecessary retracing.
If it is used by the forward function, then you must include it in
``static_argnames`` because it is used in control-flow. Ensure only ``True`` and
``False`` is passed into that parameter (avoid ``None``) so that only up to two
retracings are needed.

When jit-compiling, JAX also ignores ``None`` arguments. You can verify this by
using ``jax.make_jaxpr``.

Therefore, many ``fwd`` functions' unused ``non_trainables`` parameters will not
have a performance impact either, provided that you do not override default
behavior by passing in something other than ``None``.

Why aren't trainable and non-trainable weights Numpy arrays?
------------------------------------------------------------
Doing that would store those variables on the CPU. This would be desirable when
they do not fit on a single accelerator, as you could simply stream the model
weights and optimizer states to the appropriate accelerator as needed.

However, this comes with a performance penalty. CPU memory (RAM) tends to be
slower than accelerator memory. Allocating memory directly to an accelerator
would avoid the cost of unnecessary Host (CPU) to device (accelerator) memory
transfers. Accelerators may also initialize variables faster from an instruction
than from a memory copy. Finally, JAX performs
`Asynchronous dispatch <https://jax.readthedocs.io/en/latest/async_dispatch.html>`_,
which allows the next Python instruction to start before completing the current
JAX operation. Numpy does not perform async dispatch, which means allocating
large arrays may block the execution of the program.

In my quick benchmark on Google Colab's TPU environment, initializing two 4096
by 4096 ``float32`` JAX arrays on TPU0 and TPU1 was significantly faster than
initializing two identical numpy arrays and then transferring them to TPU0 and
TPU1.

.. figure:: images/jax_array_allocation_benchmark.jpg

    JAX arrays, which get directly allocated on accelerators, only needed 5ms.

.. figure:: images/numpy_array_allocation_benchmark.jpg

    Numpy arrays, which get initialized on the CPU before being transferred
    to the accelerators, took more than 400ms. Allocating the arrays on numpy
    alone took longer than allocating them on accelerators. Transferring the
    numpy arrays to accelerators took longer than directly allocating JAX arrays
    on accelerators as well.
    
If you wish for model weights and optimizer states to be allocated to the CPU
directly, use the `jax.default_device <https://jax.readthedocs.io/en/latest/_autosummary/jax.default_device.html>`_
context manager.

Transposed vs non-transposed Linear layer weights.
--------------------------------------------------
Transposed Linear layer weights could lead to better GEMM performance thanks to
better data locality. However, in practice, compute libraries like cuDNN have
equal or better support for GEMM with non-transposed layer weights.

AFAIK, Tensorflow uses non-transposed weights for their Linear layer and Pytorch
uses transposed weights.
`Why are weights transposed in Pytorch nn.Linear? <https://github.com/pytorch/pytorch/issues/2159>`_

Why don't Linear layers allow more flexible inputs and outputs?
-------------------------------------------------------------------
``jax.lax.dot_general`` is very powerful and allows input and output shapes to
have more than 2 dimensions. If Linear layers accept them as inputs, it would be
possible to apply linear transformations directly on Convolution layers' output
without first flattening for example.

However, using by compiling the forward function and running cost analysis on it
on both a CPU and CUDA backend, I found that reshaping the input to 2D prior
to computation and then reshaping the 2D output post-computation gave better
performance than simply using ``jax.lax.dot_general`` directly on the inputs.

Convolution kernel layouts.
---------------------------
For channel-first (NCHW), the kernel layout is OIHW. For channel-last (NHWC),
the kernel layout is OHWI. Basically, the input batch axis is replaced by the
output channel axis in the kernel, and the input channel axis remains the same.

AFAIK, this differs from Tensorflow, where the kernel layout is HWOI for both
channel-first and channel-last inputs. mlax doesn't do that is because there
could a small performance penalty. Compute libraries like cuDNN expects the
kernel to be in specific layouts and may transpose the kernel layout if needed.
`cudnnSetFilter4dDescriptor <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetFilter4dDescriptor>`_
