import jax
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

class RoPE(Module):
    """Rotary positional encoding."""
    def __init__(self, seq_len, embed_size):
        super().__init__()
        # inv_freq: (embed_size / 2,)
        # pos: (seq_len,)
        # pos_enc: (seq_len, embed_size / 2)
        inv_freq = 1.0 / ( 10000.0 ** (
            jnp.arange(0, embed_size, 2, dtype=jnp.float32) / embed_size
        ))
        pos = jnp.arange(seq_len, dtype=jnp.float32)
        pos_enc = lax.dot_general(pos, inv_freq, (((), ()), ((), ())))

        # pos_enc: (seq_len, embed_size)
        pos_enc = jnp.stack([pos_enc, pos_enc], axis=-1).reshape(
            (seq_len, embed_size)
        )

        # sin_enc, cos_enc: (seq_len, embed_size)
        self.sin = lax.sin(pos_enc)
        self.cos = lax.cos(pos_enc)

    def set_up(self, x):
        pass

    def forward(self, x, rng=None, inference_mode=False, batch_axis_name=()):
        rotated_x = jnp.stack(
            [-x[..., 1::2], x[..., ::2]], axis=-1
        ).reshape(x.shape)
        return x * self.cos + rotated_x * self.sin

class MultiQueryAttention(Module):
    """Multi-query attention."""
    def __init__(self, rng, num_heads, encode_fn, dropout_rate=0.1):
        """Initialize a multi-query attention block.

        :param rng: PRNG key.
        :param num_heads: Number of attention heads. Must divide ``q_depth``
            and ``v_depth``.
        :param encode_fn: Positional encoding function applied to query and key.
        :param dropout_rate: Dropout on attention weights.
        """
        super().__init__()

        self.rng = rng
        self.num_heads = num_heads
        self.encode_fn = encode_fn
        self.dropout_rate = dropout_rate

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.fc = None

    def set_up(self, qkvm) -> None:
        query, _, value, _ = qkvm
        q_len, q_depth = query.shape
        qk_head_depth = q_depth // self.num_heads
        _, v_depth  = value.shape
        v_head_depth = v_depth // self.num_heads

        keys_iter = iter([random.fold_in(self.rng, i) for i in range(5)])

        self.q_proj = Series([
            Linear(next(keys_iter), q_depth),
            Bias(next(keys_iter), (0, -1)),
            F(lambda x: x.reshape((q_len, self.num_heads, qk_head_depth)))
        ])

        self.k_proj = Linear(next(keys_iter), qk_head_depth)

        self.v_proj = Linear(next(keys_iter), v_head_depth)

        self.fc = Series([
            F(lambda x: x.reshape((q_len, v_depth))),
            Linear(next(keys_iter), v_depth)
        ])

    def forward(self, qkvm, rng, inference_mode=False, batch_axis_name=()):
        query, key, value, mask = qkvm

        # query: (q_length, q_depth) -> (q_length, num_heads, qk_head_depth)
        # key: (kv_length, k_depth) -> (kv_length, qk_head_depth)
        # value: (kv_length, v_depth) -> (kv_length, v_head_depth)
        # mask: broadcastable to (num_heads, q_length, kv_length)
        query, self.q_proj= self.q_proj(
            query, None, inference_mode, batch_axis_name
        )
        query = jax.vmap(self.encode_fn, in_axes=1, out_axes=1)(query)
        key, self.k_proj = self.k_proj(
            key, None, inference_mode, batch_axis_name
        )
        key = self.encode_fn(key)
        value, self.v_proj = self.v_proj(
            value, None, inference_mode, batch_axis_name
        )

        # logits: (num_heads, q_length, kv_length)
        logits = jax.vmap(
            dot_product_attention_logits, in_axes=(1, None)
        )(query, key)

        # logits, weights: (num_heads, q_length, kv_length)
        if mask is not None:
            logits = jnp.where(mask, logits, -jnp.inf)
        weights = nn.softmax(logits)

        # weights : (num_heads, q_length, kv_length)
        if inference_mode is False:
            weights = dropout(weights, rng, self.dropout_rate, (0, 1, 2))

        # activations: (q_length, num_heads, v_head_depth)
        activations = jax.vmap(
            apply_attention_weights, in_axes=(0, None), out_axes=1
        )(weights, value)

        # activations: (q_length, v_depth)
        activations, self.fc = self.fc(
            activations, None, inference_mode, batch_axis_name
        )
        return activations

class EncoderBlock(Module):
    """Multi-query encoder."""
    def __init__(
        self,
        rng,
        num_heads,
        ff_size,
        encode_fn,
        act_fn=nn.gelu,
        dropout_rate=0.1
    ):
        """ Initialize an encoder block.

        :param rng: PRNG key for weight initialization.
        :param num_heads: Number of attention heads.
        :param ff_size: Dimension of the feedforward layer.
        :param encode_fn: Positional encoding function applied to query and key.
        :param act_fn: Activation function used by the feedforward layer.
        :param dropout_rate: Dropout on attention weights and feed-forward
            activations.
        """
        super().__init__()

        self.rng = rng
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.encode_fn = encode_fn
        self.act_fn = act_fn
        self.dropout_rate = dropout_rate

        self.attention = None
        self.layer_norm1 = None
        self.layer_norm2 = None
        self.expansion = None
        self.contraction = None

    def set_up(self, xm):
        x, _ = xm
        keys_iter = iter([random.fold_in(self.rng, i) for i in range(9)])

        self.attention = MultiQueryAttention(
            next(keys_iter), self.num_heads, self.encode_fn, self.dropout_rate
        )
        self.layer_norm1 = Series([
            F(lambda x: z_norm(x, "all")),
            Scaler(next(keys_iter), (0, -1)),
            Bias(next(keys_iter), (0, -1))
        ])
        self.layer_norm2 = Series([
            F(lambda x: z_norm(x, "all")),
            Scaler(next(keys_iter), (0, -1)),
            Bias(next(keys_iter), (0, -1))
        ])
        self.expansion = Series([
            Linear(next(keys_iter), self.ff_size),
            Bias(next(keys_iter), (0, -1))
        ])
        self.contraction = Series([
            Linear(next(keys_iter), x.shape[-1]),
            Bias(next(keys_iter), (0, -1))
        ])

    def forward(self, xm, rng, inference_mode=False, batch_axis_name=()):
        # x: (seq_len, model_depth)
        # mask: (seq_len)
        x, mask = xm

        # norm_x: (seq_len, model_depth)
        norm_x, self.layer_norm1 = self.layer_norm1(
            x, None, inference_mode, batch_axis_name
        )
        # attn_weights: (seq_len, model_depth)
        attn_weights, self.attention = self.attention(
            (norm_x, norm_x, norm_x, mask),
            random.fold_in(rng, 0), inference_mode, batch_axis_name
        )
        if inference_mode is False:
            attn_weights = dropout(
                attn_weights, random.fold_in(rng, 1), self.dropout_rate, (0, 1)
            )

        # norm_attn_weights: (seq_len, model_depth)
        norm_attn_weights, self.layer_norm2 = self.layer_norm2(
            lax.add(x, attn_weights), None, batch_axis_name
        )

        # activations: (seq_len, ff_size)
        acts, self.expansion = self.expansion(
            norm_attn_weights, None, inference_mode, batch_axis_name
        )
        acts = self.act_fn(acts)
        if inference_mode is False:
            acts = dropout(
                acts, random.fold_in(rng, 2), self.dropout_rate, (0, 1)
            )
        # acts: (seq_len, model_depth)
        acts, self.contraction = self.contraction(
            acts, None, inference_mode, batch_axis_name
        )
        if inference_mode is False:
            acts = dropout(
                acts, random.fold_in(rng, 3), self.dropout_rate, (0, 1)
            )

        return lax.add(x, acts), mask
