import mlax 

def test_import():
    assert hasattr(mlax, "nn")
    assert hasattr(mlax, "functional")
