Freqently Asked Questions
=========================

How to install mlax?
--------------------
Please visit the `installation guide <https://github.com/zongyf02/mlax#installation>`_.

Is there a quickstart guide?
----------------------------
Yes! Please visit the `quick start <https://github.com/zongyf02/mlax#quickstart>`_.

Are there examples using mlax?
------------------------------
Yes! `Here on GitHub <https://github.com/zongyf02/mlax/tree/main/examples>`_.

Why should I use mlax instead of <Your Favorite ML Library>?
------------------------------------------------------------
mlax gives unparalled flexibility in developing your models! You can manage
where every single variable is stored; you can dictate how data move between
devices; you can specify where each computation happens; you can control the
precision and dtype of every single operation, all without any low-level
programming knowledge. This is mostly thanks to Google JAX's capabilities, which
also gives you jit compilation with XLA for significant speed-ups.

All of the above, along with the pure functional design, means it is easy to
develop new operations and modules in mlax (or compatible with mlax), making it
great for research and prototyping.

How to manage where variables are stored and how data flows in my model?
------------------------------------------------------------------------
First, read up on JAX's
`Controlling data and computation placement on devices <https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices>`_.

Since mlax is built with JAX, those apply to mlax model weights and optimizer
states. In short, by default, all variables are on the default devices, usually
the first accelerator. You can use `jax.device_put` to control where you put
your variables and explictly move data between devices. Also note `numpy` arrays
are always on the CPU and is transferred to the accelerator when used.

Why aren't model weights and optimizer states numpy arrays?
-----------------------------------------------------------
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

You are recommending the use of Optax's loss functions, but why not their optimizers?
-------------------------------------------------------------------------------------
First of all, if you wish to use Optax's wider selection of optimizers, you can.
You can find examples on mlax's Github.

Unlike Optax's loss functions, I feel I can meaningfully improve on Optax's
optimizers. Optimizers are complex operations; so having mlax's dtype
guarantee is important for convergence. Also, Optax does not offer guarantees
for its optimizer states' structure, which is important if you wish to partition
optimizer states across devices. Finally, Optax's optimizers cannot be jit
compiled with using the
`jax.tree_util.Partial <https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.Partial.html>`_
trick, which means using Optax's optimizers per Optax's examples leads to lower
performance than what you get with mlax's optimizers.

What is mlax's dtype guarantee?
-------------------------------
Unless explicitly overriden, mlax functions take in inputs of the same dtype,
perform all internal operations in that dtype, and outputs in that dtype. This
means no surprises during mixed-precision training.

What are the planned features? Is this project dead?
----------------------------------------------------------------
I intend to support mlax for years to come. As my learning and technology
progresse, new things will be added. On the short-term, I intend to implement
convolution, recurrent, and attention modules. As to when those features will
materialize, I cannot make guarantees as I am a full-time student.

I need <this feature>, can you add it in mlax?
----------------------------------------------
Please create an issue on GitHub, I will see what I can do!

I need <this feature>, can I add it in mlax?
--------------------------------------------
Sure, thanks! But email me first (y22zong@uwaterloo.ca), I may already be
working on it.

Who asked these freqently asked questions?
------------------------------------------
Nobody. I made them up!
