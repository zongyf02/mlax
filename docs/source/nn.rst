Neural Network Layers
=====================

``mlax.nn`` contains common neural network layers such as ``mlax.nn.Linear`` and
``mlax.nn.Conv``.

``mlax.nn`` also contains meta-layers such as ``mlax.nn.Series`` and
``mlax.nn.Parallel``, which can combine layers in series or parallel.

.. note::
    Each meta-layer has two variants. ``mlax.nn.<layer>``, which assumes its
    layers' ``__call__`` do not require a PRNGKey and whose ``__call__``
    does not require a PRNGKey, and ``mlax.nn.<layer>Rng``, which does not
    make such assumptions and whose ``__call__`` requires a PRNGKey.
    
    When possible, use the former as it is slightly more efficient.

``mlax.nn`` also contains ``mlax.nn.F`` and ``mlax.nn.FRng``, which are wrappers
that turn pure JAX functions into layers.

All of the above are implemented via ``mlax.Module``.

Finally, ``mlax.nn.functional`` contains pure JAX functions such as
``mlax.nn.functional.dropout``, ``mlax.nn.functional.max_pool``, and
``mlax.nn.functional.dot_product_attention_logits``.

.. note::   
    For maximum flexibility, all layers and functions assume their input to be
    unbatched. Use ``jax.vmap`` to add any batch dimension.
