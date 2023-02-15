import mlax

def test_import():
    assert hasattr(mlax, "Module")
    assert hasattr(mlax, "Parameter")
    assert hasattr(mlax, "fwd")
    assert hasattr(mlax, "is_mlax_module")
    assert hasattr(mlax, "is_parameter")
    assert hasattr(mlax, "is_trainable")
    assert hasattr(mlax, "is_non_trainable")
    assert hasattr(mlax, "nn")
