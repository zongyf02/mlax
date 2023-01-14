from mlax.experimental import nn

def test_nn_import():
    assert hasattr(nn, "Linear")
    assert hasattr(nn, "Bias")
    assert hasattr(nn, "F")
    assert hasattr(nn, "FRng")
    assert hasattr(nn, "Series")
    assert hasattr(nn, "SeriesRng")
    assert hasattr(nn, "Scaler")
    assert hasattr(nn, "functional")
