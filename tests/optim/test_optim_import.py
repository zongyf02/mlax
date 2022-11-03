from mlax import optim

def test_import():
    assert hasattr(optim, "sgd")
