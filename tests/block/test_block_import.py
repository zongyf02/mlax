from mlax import block

def test_import():
    assert hasattr(block, "series_fwd")
    assert hasattr(block, "series_rng_fwd")
