from mlax.experimental.nn import functional

def test_import():
    assert hasattr(functional, "identity")
    assert hasattr(functional, "dropout")
    assert hasattr(functional, "pool")
    assert hasattr(functional, "max_pool")
    assert hasattr(functional, "sum_pool")
    assert hasattr(functional, "avg_pool")
    assert hasattr(functional, "dot_product_attention_logits")
    assert hasattr(functional, "apply_attention_mask")
    assert hasattr(functional, "apply_attention_weights")
