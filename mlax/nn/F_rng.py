import jax
from typing import Tuple, Callable, Any
from mlax._utils import _nn_hyperparams

@_nn_hyperparams
class FRngHp:
    fn: Callable[[jax.Array, Any, bool], jax.Array]

def init(
    fn: Callable[[jax.Array, Any, bool], jax.Array]
) -> Tuple[None, None, FRngHp]:
    """Initialize a layer that applies an arbitrary pure functional transform 
    that consumes a PRNG key.
    
    :params fn: Pure function that takes in an input JAX array, a PRNG key, and
        a boolean indicating whether in inference or training mode, and returns
        an output JAX array.
    
    :returns trainables: None.
    :returns non_trainables: None.
    :returns hyperparams: FRngHp instance.
    """
    return None, None, FRngHp(fn)

def fwd(
    x: jax.Array,
    trainables: None,
    non_trainables: None,
    key: Any,
    hyperparams: FRngHp,
    inference_mode: bool=False
) -> jax.Array:
    """Apply an arbitrary pure functional transform that consumes a PRNG key to 
    input features.

    :param x: Input features.
    :param trainables: Trainable weights, should be None. Ignored.
    :param non_trainables: Non-trainable weights, should be None. Ignored.
    :param key: PRNG key.
    :param hyperparams: FRngHp instance.
    :param inference_mode: Whether in inference or training mode. Default:
        False, training_mode.

    :returns y: ``x`` with the arbitrary transform applied.
    :returns non_trainables: None.
    """
    return hyperparams.fn(x, key, inference_mode), None
