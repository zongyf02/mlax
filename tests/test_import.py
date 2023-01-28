import mlax

def test_import():
    assert hasattr(mlax, "Module")
    assert hasattr(mlax, "Parameter")
    assert hasattr(mlax, "is_mlax_module")
    assert hasattr(mlax, "is_parameter")
    assert hasattr(mlax, "nn")
