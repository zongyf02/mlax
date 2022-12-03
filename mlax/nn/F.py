import jax
from inspect import signature
from typing import Tuple, Callable, NamedTuple, Optional

class Hyperparams(NamedTuple):
    train_fn: Callable[[jax.Array], jax.Array]
    infer_fn: Optional[Callable[[jax.Array], jax.Array]]

def init(
    train_fn: Callable[[jax.Array], jax.Array],
    infer_fn: Optional[Callable[[jax.Array], jax.Array]] = None
) -> Tuple[None, None, Hyperparams]:
    """Initialize a layer that applies an arbitrary pure functional transform.
    
    :params train_fn: Pure function that takes in and returns a JAX array.
        Called during the forward pass in training mode.
    :params infer_fn: Optional pure function that takes in and returns a JAX
        array. Called during the forward pass in inference mode. If None, the
        ``train_fn`` is called instead. Default: None.
    
    :returns trainables: None.
    :returns non_trainables: None.
    :returns hyperparams: NamedTuple containing the hyperparameters.
    """
    return None, None, Hyperparams(train_fn, infer_fn)

def fwd(
    x: jax.Array,
    trainables: None,
    non_trainables: None,
    hyperparams: Hyperparams,
    inference_mode: bool=False
) -> jax.Array:
    """Apply an arbitrary pure functional transform on input features that does
    not require a PRNG key.

    :param x: Input features.
    :param trainables: Trainable weights, should be None. Ignored.
    :param non_trainables: Non-trainable weights, should be None. Ignored.
    :param hyperparams: NamedTuple containing the hyperparameters. 
    :param inference_mode: Whether in inference or training mode. Default:
        False, trianing mode.

    :returns y: ``x`` with the arbitrary transform applied.
    :returns non_trainables: None.
    """
    if hyperparams.infer_fn is None or not inference_mode:
        return hyperparams.train_fn(x), None
    else:
        return hyperparams.infer_fn(x), None
