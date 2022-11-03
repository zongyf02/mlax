import mlax

def test_import():
    assert hasattr(mlax, "nn")
    assert hasattr(mlax, "blocks") 
    assert hasattr(mlax, "optim")
    # Experimental is not imported by default
    assert not hasattr(mlax, "experimental")
