# MLAX: Functional NN library built on top of Google JAX

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Quickstart**](#quickstart)
| [**Examples**](https://github.com/zongyf02/mlax/tree/main/examples)
| [**Documentation**](https://mlax.readthedocs.io/en/latest/)

## What is MLAX?<a id="overview"></a>
MLAX is a purely functional neural network library built on top of Google [JAX](https://github.com/google/jax).

MLAX follows object-oriented semantics like Keras and PyTorch but remains fully
compatible with native JAX transformations.

Learn more about MLAX on [Read the Docs](https://mlax.readthedocs.io/en/latest/overview.html).

## Installation<a id="installation"></a>
[Install JAX](https://github.com/google/jax#installation) first if you have not
already.

```pip install mlax-nn```

## Quickstart<a id="quickstart"></a>
This is a simple lazy linear layer defined in MLAX.

``` Python
import jax
from jax import (
    numpy as jnp,
    nn,
    random
)
from mlax import Module, Parameter, Variable

class Linear(Module):
    def __init__(self, rng, out_features):
        super().__init__()
        self.rng = Variable(data=rng)
        self.out_features = out_features
        
        self.kernel_weight = Parameter()
        self.bias_weight = Parameter()
    
    # Define a ``set_up`` method for lazy initialziation of parameters
    def set_up(self, x):
        rng1, rng2 = random.split(self.rng.data)
        self.kernel_weight.data = nn.initializers.lecun_normal()(
            rng1, (x.shape[-1], self.out_features)
        )
        self.bias_weight.data=nn.initializers.zeros(rng2, (self.out_features,))

    # Define an ``forward`` method for the forward pass
    def forward(
        self, x, rng = None, inference_mode = False, batch_axis_name = ()
    ):
        return x @ self.kernel_weight.data + self.bias_weight.data
```

It is fully compatible with native JAX transformations:

``` Python
def loss_fn(x, y, model):
    pred, model = model(x, rng=None, inference_mode=True)
    return jnp.mean(y - pred) ** 2, model

x = jnp.ones((4, 3), dtype=jnp.float32)
y = jnp.ones((4, 4), dtype=jnp.float32)
model = Linear(random.PRNGKey(0), 4)

loss, updated_model = loss_fn(x, y, model)
print(loss)

# Now let's apply `jax.jit` and `jax.value_and_grad`
(loss, updated_model), grads = jax.jit(
    jax.value_and_grad(
        loss_fn,
        has_aux=True
    )
)(x, y, model)

print(loss)
print(grads)
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
