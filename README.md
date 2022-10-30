# [mlax]: Pure functional ML library built on top of Google [JAX]

[**Overview**](#overview)
| [**Intallation**](#installation)
| [**Quickstart**](#quickstart)
| [**Examples**](#https://github.com/zongyf02/mlax/tree/main/examples)
| **Documentation (Coming very soon)**

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
compositions of transformations. This allows for finer control of the parameters
and dataflow. This also allows [mlax] functions can be composed with [JAX]
transformations like `vmap` and `pmap` without any caveats, which makes
model-parallelism very easy to develop in [mlax].

## Sharp bits<a id="sharp-bits"></a>
[mlax] is an actively developped research project. Expect some sharp edges!

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
3. **Loss functions follow their mathematical definitions**: This may cause
unexpected behaviours. For example, `mlax.losses.categorical_crossentropy` does
not clip input predictions, which means predictions containing 0 will result in
a NaN loss. This is to allow maximum flexibility when doing loss scaling in
mixed precision training.

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
First, I recommend going through [JAX's quickstart guide](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html).
Then take a look at some [examples] (https://github.com/zongyf02/mlax/tree/main/examples).
Read the API reference here (coming soon.)

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

