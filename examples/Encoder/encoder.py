from jax import (
    lax,
    random,
    nn,
    numpy as jnp
)

from mlax import Module
from mlax.nn import Series, Linear, Bias, Scaler, F
from mlax.nn.functional import (
    dot_product_attention_logits,
    apply_attention_weights,
    z_norm,
    dropout
)

def split_heads(x, num_heads):
    return jnp.reshape(x, (x.shape[0], num_heads, -1))

def combine_heads(x):
    return jnp.reshape(x, (x.shape[0], -1))

def layer_norm(x):
    return z_norm(x, "all")

# Multihead wide attention layer
class MultiHeadedAttention(Module):
    def __init__(self, rng, num_heads, dropout_rate=0.1):
        """ Initialize a multiheaded wide attention block

        :param rng: PRNG key.
        :param qk_depth: Embedding dimension of query and key.
        :param v_depth: Embedding dimension of value.
        :param num_heads: Number of attention heads. Must divide ``qk_depth``
            and ``v_depth``.
        """
        super().__init__()

        self.rng = rng
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.fc = None

    def init(self, qkvm) -> None:
        query, key, value, _ = qkvm
        qk_depth = query.shape[-1]
        if key.shape[-1] != qk_depth:
            raise ValueError("query and key must have same depth")
        v_depth = value.shape[-1]

        keys_iter = iter([random.fold_in(self.rng, i) for i in range(8)])

        def split(x):
            return split_heads(x, self.num_heads)

        self.q_proj = Series([
            Linear(next(keys_iter), qk_depth),
            Bias(next(keys_iter), (0, -1)),
            F(split)
        ])
        self.k_proj = Series([
            Linear(next(keys_iter), qk_depth),
            Bias(next(keys_iter), (0, -1)),
            F(split)
        ])
        self.v_proj = Series([
            Linear(next(keys_iter), v_depth),
            Bias(next(keys_iter), (0, -1)),
            F(split)
        ])
        self.fc = Series([
            F(combine_heads),
            Linear(next(keys_iter), v_depth),
            Bias(next(keys_iter), (0, -1))
        ])
    
    def apply(self, qkvm, rng, inference_mode=False, batch_axis_name=()):
        query, key, value, mask = qkvm

        # query: (q_length, qk_depth) -> (q_length, num_heads, qk_head_depth)
        # key: (kv_length, qk_depth) -> (kv_length, num_heads, qk_head_depth)
        # value: (kv_length, v_depth) -> (kv_length, num_heads, v_head_depth)
        query, self.q_proj= self.q_proj(
            query, None, inference_mode, batch_axis_name
        )
        key, self.k_proj = self.k_proj(
            key, None, inference_mode, batch_axis_name
        )
        value, self.v_proj = self.v_proj(
            value, None, inference_mode, batch_axis_name
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
        if inference_mode is False:
            weights = dropout(weights, rng, self.dropout_rate, 2)

        # activations: (q_length, num_heads, v_head_depth)
        activations = apply_attention_weights(value, weights)

        # activations: (q_length, v_head_depth)
        activations, self.fc = self.fc(
            activations, None, inference_mode, batch_axis_name
        )
        return activations

# Encoder layer with lazy initialization
class EncoderBlock(Module):
    def __init__(
        self,
        rng,
        num_heads=8,
        ff_depth=1024,
        act_fn=nn.gelu,
        dropout_rate=0.1
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

        self.rng = rng
        self.num_heads = num_heads
        self.ff_depth = ff_depth
        self.act_fn = act_fn
        self.dropout_rate = dropout_rate

        self.attention = None
        self.layer_norm1 = None
        self.layer_norm2 = None
        self.expansion = None
        self.contraction = None

        self.dropout_rate = dropout_rate

    def init(self, xm):
        x, _ = xm

        keys_iter = iter([random.fold_in(self.rng, i) for i in range(9)])

        self.attention = MultiHeadedAttention(
            next(keys_iter), self.num_heads, self.dropout_rate
        )
        self.layer_norm1 = Series([
            F(layer_norm),
            Scaler(next(keys_iter), (0, -1)),
            Bias(next(keys_iter), (0, -1))
        ])
        self.layer_norm2 = Series([
            F(layer_norm),
            Scaler(next(keys_iter), (0, -1)),
            Bias(next(keys_iter), (0, -1))
        ])
        self.expansion = Series([
            Linear(next(keys_iter), self.ff_depth),
            Bias(next(keys_iter), (0, -1))
        ])
        self.contraction = Series([
            Linear(next(keys_iter), x.shape[-1]),
            Bias(next(keys_iter), (0, -1))
        ])

    def apply(self, xm, rng, inference_mode=False, batch_axis_name=()):
        x, mask = xm
        keys_iter = iter([random.fold_in(self.rng, i) for i in range(4)])

        # x: (seq_len, model_depth)
        # mask: (num_heads, seq_len, seq_len)
        x, self.layer_norm1 = self.layer_norm1(
            x, None, inference_mode, batch_axis_name
        )

        # attention_weights: (seq_len, model_depth)
        attention_weights, self.attention = self.attention(
            (x, x, x, mask), next(keys_iter), inference_mode, batch_axis_name
        )
        if inference_mode is False:
            attention_weights = dropout(
                attention_weights, next(keys_iter), self.dropout_rate, axis=1
            )
        
        # x: (seq_len, model_depth)
        x, self.layer_norm2 = self.layer_norm2(
            lax.add(x, attention_weights), None, batch_axis_name
        )

        # forward: (seq_len, model_depth)
        forward, self.expansion = self.expansion(
            x, None, inference_mode, batch_axis_name
        )
        if inference_mode is False:
            forward = dropout(
                self.act_fn(forward), next(keys_iter), self.dropout_rate, axis=1
            )
        forward, self.contraction = self.contraction(
            forward, None, inference_mode, batch_axis_name
        )
        if inference_mode is False:
            forward = dropout(
                forward, next(keys_iter), self.dropout_rate, axis=1
            )
        return lax.add(x, forward)
