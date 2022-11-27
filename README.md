# [mlax]: Pure functional ML library built on top of Google [JAX]

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Quickstart**](#quickstart)
| [**Examples**](https://github.com/zongyf02/mlax/tree/main/examples)
| [**Documentation**](https://mlax.readthedocs.io/en/latest/)

## What is [mlax]?<a id="overview"></a>
[mlax] is a ML library built with Google [JAX], and it follows [JAX]'s
[pure functional paradigm](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions).

This means [mlax] is fully compatible with native JAX transformations; and you
don't need anything else for [mlax] to work.

If you understand [JAX], you understand [mlax]!

## Installation<a id="installation"></a>
[mlax] is on PyPi. You can install [mlax] using `pip`.

```pip install mlax-nn```

Note that this also installs the CPU version of JAX on your machine. If you need
GPU acceleration, follow JAX's [installation guide](https://github.com/google/jax#installation).

## Quickstart<a id="quickstart"></a>
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
[Tensorflow]: https://www.tensorflow.org/
[Pytorch]: https://pytorch.org/
