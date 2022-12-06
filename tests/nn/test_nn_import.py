from mlax import nn 

def test_import():
    assert hasattr(nn, "Linear")
    assert hasattr(nn, "Bias")
    assert hasattr(nn, "Scaler")
    assert hasattr(nn, "Conv")
    assert hasattr(nn, "F")
    assert hasattr(nn, "F_rng")
    assert hasattr(nn, "BatchNorm")
