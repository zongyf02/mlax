# mlax: Pure functional ML library built on top of Google [JAX]

## What is mlax?
mlax is a ML library built with [JAX], and it follows [JAX]'s
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



