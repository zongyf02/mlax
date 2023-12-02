import mlax

def test_import():
    assert hasattr(mlax, "Parameter")
    assert hasattr(mlax, "Variable")
    assert hasattr(mlax, "Container")
    assert hasattr(mlax, "Module")
    assert hasattr(mlax, "is_leaf_state")
    assert hasattr(mlax, "is_variable")
    assert hasattr(mlax, "is_parameter")
    assert hasattr(mlax, "is_frozen_param")
    assert hasattr(mlax, "is_unfrozen_param")
    assert hasattr(mlax, "is_container")
    assert hasattr(mlax, "is_module")
    assert hasattr(mlax, "nn")
