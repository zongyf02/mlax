# [mlax]: Pure functional ML library built on top of Google [JAX]

[**Overview**](#overview)
| [**Intallation**](#installation)
| [**Quickstart**](#quickstart)
| [**Examples**](https://github.com/zongyf02/mlax/tree/main/examples)
| [**Documentation**](https://mlax.readthedocs.io/en/latest/)

## What is [mlax]?<a id="overview"></a>
[mlax] is a ML library built with Google [JAX], and it follows [JAX]'s
[pure functional paradigm](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions).

This means there are no layers, mutations, nor classes. Models are a series of
transformations without internal states or side-effects.

## Why functional programming?
Pure functional programs are easy to optimize and parallelize. [JAX] relies on
functional programming do optimize Python functions with `jit` and parallelize
transformations with `pmap`.

## Why use [mlax] over [flax] or [haiku]?
[flax] and [haiku] are also excellent ML libraries built on top of [JAX]. They
are maintained respectively by Google and DeepMind. They use a module-based
programming model, which should be familiar to anyone coming from [Tensorflow] 
or [Pytorch].

On the other hand, [mlax] favors function-based programming. Models are simply
compositions of pure functions, which allows [mlax] to be used with [JAX]
without any caveats. Thanks to [JAX]'s `vmap` and `pmap`, model-parallelism
is easy and intuitive with [mlax].

[mlax] also offers strong dtype guarantees. Unless explicitly overriden, [mlax]
functions takes in inputs of the same dtype, perform all internal operations in
the that dtype, and ouputs the same type. This means no surprises during
mixed-precision training.

## Sharp bits<a id="sharp-bits"></a>
[mlax] is an actively developed research project. Expect some sharp bits!

In addition to [JAX's sharp bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html),
here are some behaviors of [mlax] that differ from other ML libraries:
1.  **No implicit type promotion**: [mlax] functions do not implicily cast its
parameters. Unless explicitly stated, Array arguemnts must be of the same dtype.
This is to avoid surprises when doing mixed precision training.
2. **Functions do not support batched inputs**: [mlax] functions assume that
their input is a single unbatched sample. To get functions that work on batched
inputs, use [JAX]'s `vmap` or `pmap` transformations. This is to offer more
flexibility when parallelizing models and when handling different bactch
dimensions.

## Installation<a id="installation"></a>
[mlax] is written in pure Python. You can install [mlax] using `pip`.

```pip install mlax-nn```

You can also install from source. First clone the repository:

```git clone https://github.com/zongyf02/mlax.git```

Move into the cloned repository, then build a Python wheel.

```python3 setup.py bdist_wheel```

Finally, install the wheel locally.

```pip install -e mlax-nn```

## Quickstart<a id="quickstart"></a>
Before you start, I recommend going through JAX's
[quickstart guide](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html),
and the [worked example on stateful computations](https://jax.readthedocs.io/en/latest/jax-101/07-state.html#simple-worked-example-linear-regression).

mlax relies on Optax to update its weights, read its [Quick Start](https://optax.readthedocs.io/en/latest/optax-101.html) as well.

Then take a look at mlax's [API Overview](https://mlax.readthedocs.io/en/latest/overview.html).

Run some [examples](https://github.com/zongyf02/mlax/tree/main/examples) with
reference implementations in [Pytorch].

Finally, read the [API Reference](https://mlax.readthedocs.io/en/latest/apidocs/modules.html).

## Bugs and Feature Requests
Please [create an issue](https://github.com/zongyf02/mlax/issues) on [mlax]'s
Github repository.

## Contribution
If you wish to contribute, thank you and please contact me by email:
y22zong@uwaterloo.ca.

[mlax]: https://github.com/zongyf02/mlax
[JAX]: https://github.com/google/jax
[flax]: https://github.com/google/flax
[haiku]: https://github.com/deepmind/dm-haiku
[Tensorflow]: https://www.tensorflow.org/
[Pytorch]: https://pytorch.org/

