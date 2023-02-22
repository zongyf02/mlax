import jax
from jax import (
    lax,
    random,
    nn,
    numpy as jnp
)

from mlax import Module, Parameter
from mlax.nn import Series, Linear, Bias, Scaler, F
from mlax.nn.functional import (
    dot_product_attention_logits,
    apply_attention_weights,
    layer_norm,
    dropout
)

# Multihead wide attention layer
class MultiHeadedAttention(Module):
    def __init__(self, rng, qk_depth, v_depth, num_heads, dropout=0.1):
        """ Initialize a multiheaded wide attention block

        :param: PRNG key for weight initialization.
        :param qk_depth: Embedding dimension of query and key.
        :param v_depth: Embedding dimension of value.
        :param num_heads: Number of attention heads. Must divide ``qk_depth``
            and ``v_depth``.
        """
        super().__init__()
        rngs_iter = iter(random.split(rng, 8))

        def split_heads(x):
            return jnp.reshape(x, (x.shape[0], num_heads, -1))
        
        def combine_heads(x):
            return jnp.reshape(x, (x.shape[0], -1))

        self.q_linear = Series([
            Linear(next(rngs_iter), qk_depth),
            Bias(next(rngs_iter), (None, -1)),
            F(split_heads)
        ])

        self.k_linear = Series([
            Linear(next(rngs_iter), qk_depth),
            Bias(next(rngs_iter), (None, -1)),
            F(split_heads)
        ])

        self.v_linear = Series([
            Linear(next(rngs_iter), v_depth),
            Bias(next(rngs_iter), (None, -1)),
            F(split_heads)
        ])

        self.fc_out = Series([
            F(combine_heads),
            Linear(next(rngs_iter), v_depth),
            Bias(next(rngs_iter), (None, -1))
        ])

        self.dropout = dropout
    
    def __call__(
        self,
        x,
        rng,
        inference_mode=False
    ):
        """ Apply multiheaded attention to query, key, value, and mask.

        :param x: Tuple of JAX arrays containing query of shape
            ``(q_length, qk_depth)``, key of shape
            ``(kv_length, qk_depth)``, value of shape
            ``(kv_length, v_depth)``, and mask broadcastable to shape
            ``(num_heads, q_length, kv_length)``.
        :param rng: PRNG key.
        :param inference_mode: Whether in inference or training mode. Ignored.
            Default: False.

        :return x: Output of shape ``(q_length, v_depth)``.
        :return self: MultiHeadedAttention layer.

        .. note:
            Query, key, and value do not have to have ``qk_depth`` or
            ``v_depth`` as their last dimension because a linear layer is
            applied before scaled dot-product attention.
        """
        query, key, value, mask = x

        # query: (q_length, qk_depth) -> (q_length, num_heads, qk_head_depth)
        # key: (kv_length, qk_depth) -> (kv_length, num_heads, qk_head_depth)
        # value: (kv_length, v_depth) -> (kv_length, num_heads, v_head_depth)
        query, self.q_linear = self.q_linear(
            query, None, inference_mode=inference_mode
        )
        key, self.k_linear = self.k_linear(
            key, None, inference_mode=inference_mode
        )
        value, self.v_linear = self.v_linear(
            value, None, inference_mode=inference_mode
        )

        # logits: (num_heads, q_length, kv_length)
        logits = dot_product_attention_logits(query, key)

        # # logits: (num_heads, q_length, kv_length)
        if mask is not None:
            logits = jnp.where(
                mask, logits, lax.convert_element_type(-jnp.inf, logits.dtype)
            )

        # weights: (num_heads, q_length, kv_length)
        weights = nn.softmax(logits)
        if inference_mode:
            weights = jax.vmap(
                jax.vmap(dropout, in_axes=[0, None, None]),
                in_axes=[0, None, None]
            )(weights, rng, self.dropout)

        # activations: (q_length, num_heads, v_head_depth)
        activations = apply_attention_weights(value, weights)

        # activations: (q_length, v_head_depth)
        activations, self.fc_out = self.fc_out(
            activations, None, inference_mode=inference_mode
        )
        return activations, self

# Encoder layer with lazy initialization
class EncoderBlock(Module):
    def __init__(
        self,
        rng,
        model_depth,
        num_heads=8,
        ff_depth=1024,
        act_fn=nn.gelu,
        dropout=0.1
    ):
        """ Initialize an encoder block.

        :param rng: PRNG key for weight initialization.
        :param model_depth: Embedding dimension of query, key, and value.
        :param num_heads: Number of attention heads.
        :param ff_depth: Dimension of the feedforward layer.
        :param act_fn: Activation function used by the feedforward layer.
            Default: GELU.
        :param dropout: Dropout rate. Default: 0.1.
        """
        super().__init__()
        
        rngs_iter = iter(random.split(rng, 8))
        self._rng = Parameter(trainable=False, data=next(rngs_iter))

        self.attention = MultiHeadedAttention(
            next(rngs_iter), model_depth, model_depth, num_heads, dropout
        )

        self.layer_norm1 = Series([
            F(jax.vmap(layer_norm)),
            Scaler(next(rngs_iter), (None, -1)),
            Bias(next(rngs_iter), (None, -1)),
        ])

        self.layer_norm2 = Series([
            F(jax.vmap(layer_norm)),
            Scaler(next(rngs_iter), (None, -1)),
            Bias(next(rngs_iter), (None, -1)),
        ])
        
        self.expansion = Series([
            Linear(next(rngs_iter), ff_depth),
            Bias(next(rngs_iter), (None, -1)),
            F(act_fn)
        ])
        self.contraction = None

        self.dropout = dropout

        self.initialized = False

    def _build(self, query):
        """Initialize an uninitialized batch norm layer."""
        rng1, rng2 = random.split(self._rng.data)
        self.contraction = Series([
            Linear(rng1, query.shape[-1]),
            Bias(rng2, (None, -1))
        ])
        del self._rng

    def __call__(self, x, rng, inference_mode=False):
        """Apply encoder to query, key, value, and mask features.

        :param x: JAX arrays containing embeddings of shape
            ``(seq_len, model_depth)`` and mask broadcastable to shape
            ``(num_heads, seq_len, seq_len)``.
        :param rng: PRNG key.
        :param inference_mode: Whether in inference or training mode.
            Default: False.

        :return x: Output of shape ``(q_length, v_depth)``.
        :return self: EncoderBlock layer.
        """

        # embeddings: (seq_len, model_depth)
        # mask: (num_heads, seq_len, seq_len)
        embeddings, mask = x
        if not self.initialized:
            self._build(embeddings)
            self.initialized = True

        rng1, rng2, rng3 = random.split(rng, 3)

        # attention_weights: (seq_len, model_depth)
        attention_weights, self.attention = self.attention(
            (embeddings, embeddings, embeddings, mask), rng1, inference_mode
        )
        if not inference_mode:
            attention_weights = dropout(attention_weights, rng2, self.dropout)

        # embeddings: (seq_len, model_depth)
        embeddings, self.layer_norm1 = self.layer_norm1(
            lax.add(embeddings, attention_weights)
        )

        # forward: (seq_len, model_depth)
        forward, self.expansion = self.expansion(
            embeddings, None, inference_mode
        )
        if not inference_mode:
            forward = dropout(forward, rng2, self.dropout)
        forward, self.contraction = self.contraction(
            forward, None, inference_mode
        )
        if not inference_mode:
            forward = dropout(forward, rng2, self.dropout)

        # embeddings: (seq_len, model_depth)
        embeddings, self.layer_norm2 = self.layer_norm2(
            lax.add(embeddings, forward)
        )
        return embeddings, self
