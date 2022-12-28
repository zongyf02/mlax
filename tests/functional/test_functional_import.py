from mlax import functional

def test_import():
    assert hasattr(functional, "identity")
    assert hasattr(functional, "dropout")
    assert hasattr(functional, "pool")
    assert hasattr(functional, "max_pool")
    assert hasattr(functional, "sum_pool")
    assert hasattr(functional, "avg_pool")
    assert hasattr(functional, "layer_norm")
    assert hasattr(functional, "instance_norm")
    assert hasattr(functional, "group_norm")
