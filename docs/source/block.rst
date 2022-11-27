``mlax.block``
===============

The ``mlax.block`` package contains functions that combine multiple layers.
Examples include ``mlax.block.series_fwd`` and ``mlax.block.series_rng_fwd``,
which combines the layers in series.

There are always two versions of a same type of block function:

* A function with ``_rng`` in its name, which only works on layers that do not
    consumme a PRNG key.
* A function with ``_rng`` in its name, which consummes a PRNG key, allowing it
    to work on layers requiring PRNG keys as well. However, splitting PRNG keys
    comes at a slight performance cost, so the first option is preferred
    whenever possible.
