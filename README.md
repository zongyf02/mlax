# MLAX: Pure functional ML library built on top of Google JAX

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Quickstart**](#quickstart)
| [**Examples**](https://github.com/zongyf02/mlax/tree/main/examples)
| [**Documentation**](https://mlax.readthedocs.io/en/latest/)

## What is MLAX?<a id="overview"></a>
MLAX is a purely functional ML library built on top of Google [JAX](https://github.com/google/jax).

MLAX follows object-oriented semantics like Keras and PyTorch but remains fully
compatible with native JAX transformations.

Learn more about MLAX on [Read the Docs](https://mlax.readthedocs.io/en/latest/overview.html).

## Installation<a id="installation"></a>
[Install JAX](https://github.com/google/jax#installation) first if you have not
already.

```pip install mlax-nn```

## Quickstart<a id="quickstart"></a>
This is a simple linear layer defined using only the MLAX Module and Parameter.

``` Python
import jax
from jax import (
    numpy as jnp,
    nn,
    random
)
from mlax import Module, Parameter

class Linear(Module):
    def __init__(self, in_features, out_features, rng):
        rng1, rng2 = random.split(rng)
        self.kernel_weight = Parameter(
            trainable=True,
            data=nn.initializers.glorot_uniform()(rng1, (in_features, out_features))
        )
        self.bias_weight = Parameter(
            trainable=True,
            data=nn.initializers.zeros(rng2, (out_features,))
        )
    
    def __call__(self, x, rng=None, inference_mode=False):
        return x @ self.kernel_weight.data + self.bias_weight.data, self
```

It is fully compatible with native JAX transformations:

``` Python
def loss_fn(model, x, y):
    pred, model = model(x)
    return jnp.mean(y - pred) ** 2, model

model = Linear(3, 4, random.PRNGKey(0))
x = jnp.ones((4, 3), dtype=jnp.float32)
y = jnp.ones((4, 4), dtype=jnp.float32)

(loss, model), grads = jax.jit(
    jax.value_and_grad(
        loss_fn,
        has_aux=True
    )
)(model, x, y)
```

For end-to-end examples with reference PyTorch implementations, visit MLAX's
[GitHub](https://github.com/zongyf02/mlax/tree/main/examples).

View the full documentation on [Read the Docs](https://mlax.readthedocs.io/en/latest/).

## Bugs and Feature Requests
Please [create an issue](https://github.com/zongyf02/mlax/issues) on MLAX's
Github repository.

## Contribution
If you wish to contribute, thank you and please contact me by email:
y22zong@uwaterloo.ca.
