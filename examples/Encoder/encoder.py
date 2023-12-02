import jax
from jax import (
    lax,
    random,
    nn,
    numpy as jnp
)
from mlax import Variable, Module
from mlax.nn import Series, Linear, Bias, Scaler, F
from mlax.nn.functional import (
    rms_norm,
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
        self.sin = Variable(data=lax.sin(pos_enc))
        self.cos = Variable(data=lax.cos(pos_enc))

    def set_up(self, x):
        pass

    def forward(self, x, rng=None, inference_mode=False, batch_axis_name=()):
        rotated_x = jnp.stack(
            [-x[..., 1::2], x[..., ::2]], axis=-1
        ).reshape(x.shape)
        return x * self.cos.data + rotated_x * self.sin.data

class MultiheadAttention(Module):
    """Multi-head attention."""
    def __init__(self, rng, num_heads, encode_fn, attention_fn):
        """Initialize a multi-head attention block.

        :param rng: PRNG key.
        :param num_heads: Number of attention heads. Must divide ``q_depth``
            and ``v_depth``.
        :param encode_fn: Positional encoding function applied to query and key.
        :param attention_fn: Attention function applied to query, key, and mask.
        """
        super().__init__()

        self.rng = Variable(data=rng)
        self.num_heads = num_heads
        self.encode_fn = encode_fn
        self.attention_fn = attention_fn

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.fc = None

    def set_up(self, qkvm) -> None:
        query, _, value, _ = qkvm
        q_len, qk_depth = query.shape
        qk_head_depth = qk_depth // self.num_heads
        kv_len, v_depth  = value.shape
        v_head_depth = v_depth // self.num_heads

        keys_iter = iter([random.fold_in(self.rng.data, i) for i in range(4)])

        self.q_proj = Series([
            Linear(next(keys_iter), qk_depth),
            F(lambda x: x.reshape((q_len, self.num_heads, qk_head_depth)))
        ])
        self.k_proj = Series([
            Linear(next(keys_iter), qk_depth),
            F(lambda x: x.reshape((kv_len, self.num_heads, qk_head_depth)))
        ])
        self.v_proj = Series([
            Linear(next(keys_iter), v_depth),
            F(lambda x: x.reshape((kv_len, self.num_heads, v_head_depth)))
        ])

        self.fc = Series([
            F(lambda x: x.reshape((q_len, v_depth))),
            Linear(next(keys_iter), v_depth)
        ])

    def forward(self, qkvm, rng=None, inference_mode=False, batch_axis_name=()):
        query, key, value, mask = qkvm

        # query: (q_length, qk_depth) -> (q_length, num_heads, qk_head_depth)
        # key: (kv_length, qk_depth) -> (kv_length, num_heads, qk_head_depth)
        # value: (kv_length, v_depth) -> (kv_length, num_heads, v_head_depth)
        # mask: (num_heads, q_length, kv_length)
        query, self.q_proj= self.q_proj(
            query, None, inference_mode, batch_axis_name
        )
        query = jax.vmap(self.encode_fn, in_axes=1, out_axes=1)(query)
        key, self.k_proj = self.k_proj(
            key, None, inference_mode, batch_axis_name
        )
        query = jax.vmap(self.encode_fn, in_axes=1, out_axes=1)(query)
        value, self.v_proj = self.v_proj(
            value, None, inference_mode, batch_axis_name
        )
        mask = jnp.broadcast_to(
            mask, (self.num_heads, query.shape[0], key.shape[0])
        )

        # activations: (q_length, num_heads, v_head_depth)
        activations = self.attention_fn(query, key, value, mask)

        # activations: (q_length, v_depth)
        activations, self.fc = self.fc(
            activations, None, inference_mode, batch_axis_name
        )
        return activations

class SwiGLU(Module):
    """SwiGLU activation."""
    def __init__( self, rng, hidden_size):
        """Initialize a SwiGLU activation.
        
        :param rng: PRNG key for weight initialization.
        :param hidden_size: Intermediate dimension.
        """
        super().__init__()

        self.rng = Variable(data=rng)
        self.hidden_size = hidden_size

        self.w1 = None
        self.w2 = None
        self.w3 = None
    
    def set_up(self, x):
        keys_iter = iter([random.fold_in(self.rng.data, i) for i in range(6)])
        self.w1 = Series([
            Linear(next(keys_iter), self.hidden_size),
            Bias(next(keys_iter), (0, -1))
        ])
        self.w2 = Series([
            Linear(next(keys_iter), x.shape[-1]),
            Bias(next(keys_iter), (0, -1))
        ])
        self.w3 = Series([
            Linear(next(keys_iter), self.hidden_size),
            Bias(next(keys_iter), (0, -1))
        ])

    def forward(self, x, rng=None, inference_mode=False, batch_axis_name=()):
        x1, self.w1 = self.w1(x, rng, inference_mode, batch_axis_name)
        x2, self.w3 = self.w3(x, rng, inference_mode, batch_axis_name)
        y, self.w2 = self.w2(
            lax.mul(nn.silu(x1), x2), rng, inference_mode, batch_axis_name
        )
        return y

class EncoderBlock(Module):
    """Multi-head encoder."""
    def __init__(
        self,
        rng,
        num_heads,
        ff_size,
        encode_fn,
        attention_fn,
        dropout_rate=0.1
    ):
        """Initialize an encoder block.

        :param rng: PRNG key for weight initialization.
        :param num_heads: Number of attention heads.
        :param ff_size: Dimension of the feedforward layer.
        :param encode_fn: Positional encoding function applied to query and key.
        :param attention_fn: Attention function applied on query, key, values,
            and mask.
        :param dropout_rate: Dropout on attention weights and feed-forward
            activations.
        """
        super().__init__()

        self.rng = Variable(data=rng)
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.encode_fn = encode_fn
        self.attention_fn = attention_fn
        self.dropout_rate = dropout_rate

        keys_iter = iter([random.fold_in(self.rng.data, i) for i in range(4)])

        self.attention = MultiheadAttention(
            next(keys_iter), self.num_heads, self.encode_fn, self.attention_fn
        )
        self.rms_norm1 = Series([
            F(lambda x: rms_norm(x, 1)),
            Scaler(next(keys_iter), (0, -1))
        ])
        self.rms_norm2 = Series([
            F(lambda x: rms_norm(x, 1)),
            Scaler(next(keys_iter), (0, -1))
        ])
        self.swiglu = SwiGLU(next(keys_iter), self.ff_size)

    def set_up(self, xm):
        pass

    def forward(self, xm, rng, inference_mode=False, batch_axis_name=()):
        # x: (seq_len, model_depth)
        # mask: (seq_len)
        x, mask = xm

        # norm_x: (seq_len, model_depth)
        norm_x, self.rms_norm1 = self.rms_norm1(
            x, None, inference_mode, batch_axis_name
        )
        # attn_weights: (seq_len, model_depth)
        attn_weights, self.attention = self.attention(
            (norm_x, norm_x, norm_x, mask),
            None, inference_mode, batch_axis_name
        )
        if inference_mode is False:
            attn_weights = dropout(
                attn_weights, random.fold_in(rng, 0), self.dropout_rate, (0, 1)
            )

        # norm_attn_weights: (seq_len, model_depth)
        norm_attn_weights, self.rms_norm2 = self.rms_norm2(
            lax.add(x, attn_weights), None, batch_axis_name
        )

        # acts: (seq_len, model_depth)
        acts, self.swiglu = self.swiglu(
            norm_attn_weights, None, inference_mode, batch_axis_name
        )
        if inference_mode is False:
            acts = dropout(
                acts, random.fold_in(rng, 1), self.dropout_rate, (0, 1)
            )

        return lax.add(x, acts), mask
