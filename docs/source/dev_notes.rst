Dev Notes
==========

Planned features ranked by priority
------------------------------------
Batchnorm, attention, recurrent layers.

Why aren't trainable and non-trainble weights Numpy arrays?
------------------------------------------------------------
Doing that would store those variables on the CPU. This would be desirable when
they do not fit on a single accelerator, as you could simply stream the model
weights and optimizer states to the appropriate accelerator as needed.

However, this comes with a performance penalty. CPU memory (RAM) tends to be
slower than accelerator memory. Allocating memory directly on an accelerator
would avoid the cost of unnecessary Host (CPU) to device (accelerator) memory
transfers. Accelerators may also initialize variables faster from an instruction
than from a memory copy. Finally, JAX performs
`Asynchronous dispatch <https://jax.readthedocs.io/en/latest/async_dispatch.html>`_,
which allows the next Python instruction to start before completing the current
JAX operation. Numpy does not perform async dispatch, which means allocating
large arrays may block the execution of the program.

In my quick benchmark on Google Colab's TPU environment, initializing two 4096
by 4096 ``float32`` JAX arrays on TPU0 and TPU1 was significantly faster than
initializing two identical numpy arrays then transferring them to TPU0 and TPU1.

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