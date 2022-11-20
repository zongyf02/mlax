from mlax import nn 

def test_import():
    assert hasattr(nn, "linear")
    assert hasattr(nn, "bias")
    assert hasattr(nn, "dropout")
    assert hasattr(nn, "conv_f")
    assert hasattr(nn, "conv_l")
