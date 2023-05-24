import mlax

def test_import():
    assert hasattr(mlax, "Parameter")
    assert hasattr(mlax, "is_trainable_param")
    assert hasattr(mlax, "is_non_trainable_param")
    assert hasattr(mlax, "is_leaf_param")
    assert hasattr(mlax, "Module")
