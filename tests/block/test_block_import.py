from mlax import block

def test_import():
    assert hasattr(block, "Series")
    assert hasattr(block, "Series_rng")
    assert hasattr(block, "Parallel")
    assert hasattr(block, "Parallel_rng")
