from mlax import nn

def test_nn_import():
    assert hasattr(nn, "Linear")
    assert hasattr(nn, "Bias")
    assert hasattr(nn, "F")
    assert hasattr(nn, "FRng")
    assert hasattr(nn, "Series")
    assert hasattr(nn, "SeriesRng")
    assert hasattr(nn, "Scaler")
    assert hasattr(nn, "Conv")
    assert hasattr(nn, "ZNorm")
    assert hasattr(nn, "Parallel")
    assert hasattr(nn, "ParallelRng")
    assert hasattr(nn, "Embed")
    assert hasattr(nn, "Recurrent")
    assert hasattr(nn, "RecurrentRng")
