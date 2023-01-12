import jax
from mlax.experimental import Parameter, Module

class Bar(Module):
    def __init__(self):
        self.x = Parameter(True, 0)
        self.y = Parameter(False, 1)
        self.z = "a"
    
    def __call__(self, xy, key=None, inference_mode=False):
        x, y = xy
        self.x.data = x
        self.y.data = y
        return xy, self

b = Bar()
trainables = b.trainables
non_trainables = b.non_trainables

def test_load():
    # Assert correct trainables and non_trainables
    assert len(trainables) == 1
    assert trainables[0].data == 0
    assert len(non_trainables) == 1
    assert non_trainables[0].data == 1

    # Assert seperate copies of trainables and non_trainables
    trainables[0].data = 1
    non_trainables[0].data = 2
    assert b.x.data == 0
    assert b.y.data == 1

    # Assert correct load
    b_reconstructed = b.load_trainables(
        trainables
    ).load_non_trainables(
        non_trainables
    )
    assert b_reconstructed.x.data == 1
    assert b_reconstructed.y.data == 2
    assert b_reconstructed.z == "a"

    # Assert independent loaded copy
    b_reconstructed is not b # Not the same object
    b_reconstructed.x.data = 2
    b_reconstructed.y.trainable = True
    b_reconstructed.z = 0
    assert b.x.data == 0
    assert b.y.trainable == False
    assert b.z == "a"

def test_call():
    xy = (5, 6)
    new_xy, new_b = b(xy)
    assert new_xy == xy
    assert new_b is b # Same object
    assert new_b.x.data == 5
    assert new_b.y.data == 6
    assert new_b.z == "a"

def test_fwd():
    xy = (8, 9)
    new_xy, new_b = jax.jit(
        Bar.fwd, static_argnames="inference_mode"
    )(
        b,
        b.trainables,
        xy,
        jax.random.PRNGKey(0),
        inference_mode=False
    )
    assert xy == xy
    assert new_b is not b # Not the same object
    assert new_b.x.data == 8
    assert new_b.y.data == 9
    assert new_b.z == "a"