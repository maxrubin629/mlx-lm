# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_causal_mask, scaled_dot_product_attention
from .cache import CacheList, KVCache, RotatingKVCache
from .rope_utils import initialize_rope


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "t5gemma2_text"
    hidden_size: int = 640
    num_hidden_layers: int = 18
    intermediate_size: int = 2048
    num_attention_heads: int = 4
    head_dim: int = 256
    rms_norm_eps: float = 1.0e-6
    vocab_size: int = 262144
    num_key_value_heads: int = 1
    query_pre_attn_scalar: float = 256
    sliding_window: int = 512
    max_position_embeddings: int = 32768
    rope_parameters: Optional[Dict[str, Dict[str, Any]]] = None
    layer_types: Optional[list[str]] = None
    _sliding_window_pattern: int = 6
    hidden_activation: str = "gelu_pytorch_tanh"
    attention_bias: bool = False
    attn_logit_softcapping: Optional[float] = None
    final_logit_softcapping: Optional[float] = None
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 1

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if (i + 1) % self._sliding_window_pattern
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default",
                },
                "full_attention": {
                    "rope_theta": 1000000.0,
                    "rope_type": "linear",
                    "factor": 8.0,
                },
            }

        self.rope_parameters.setdefault(
            "sliding_attention",
            {"rope_theta": 10000.0, "rope_type": "default"},
        )
        self.rope_parameters.setdefault(
            "full_attention",
            {"rope_theta": 1000000.0, "rope_type": "linear", "factor": 8.0},
        )


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "t5gemma2"
    encoder: Optional[dict] = None
    decoder: Optional[dict] = None
    vocab_size: int = 262144
    eoi_token_index: int = 256000
    image_token_index: int = 256001
    bos_token_id: int = 2
    pad_token_id: int = 0
    eos_token_id: int = 1

    def __post_init__(self):
        self.encoder = self.encoder or {}
        self.decoder = self.decoder or {}

        if "text_config" in self.encoder:
            self.encoder_text_config = dict(self.encoder["text_config"])
        else:
            self.encoder_text_config = dict(self.encoder)

        self.decoder_config = dict(self.decoder)
        self.encoder_text_config["vocab_size"] = self.vocab_size
        self.decoder_config["vocab_size"] = self.vocab_size

        self.encoder_text_config.setdefault("model_type", "t5gemma2_text")
        self.decoder_config.setdefault("model_type", "t5gemma2_decoder")
        self.encoder_text_config.setdefault("bos_token_id", self.bos_token_id)
        self.decoder_config.setdefault("bos_token_id", self.bos_token_id)
        self.encoder_text_config.setdefault("pad_token_id", self.pad_token_id)
        self.decoder_config.setdefault("pad_token_id", self.pad_token_id)
        self.encoder_text_config.setdefault("eos_token_id", self.eos_token_id)
        self.decoder_config.setdefault("eos_token_id", self.eos_token_id)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class ScaledEmbedding(nn.Module):
    def __init__(self, args: TextArgs, eoi_token_index: int):
        super().__init__()
        self.weight = mx.zeros((args.vocab_size, args.hidden_size))
        self.eoi_embedding = mx.zeros((args.hidden_size,))
        self.scale = args.hidden_size**0.5
        self.eoi_token_index = eoi_token_index

    def __call__(self, x: mx.array) -> mx.array:
        out = self.weight[x] * self.scale
        if self.eoi_token_index is not None:
            out = mx.where(
                (x == self.eoi_token_index)[..., None],
                self.eoi_embedding.astype(out.dtype),
                out,
            )
        return out

    def as_linear(self, x: mx.array) -> mx.array:
        return x @ self.weight.T

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        quantized = nn.Embedding(self.weight.shape[0], self.weight.shape[1])
        quantized.weight = self.weight
        return quantized.to_quantized(group_size=group_size, bits=bits, mode=mode)


def _attention_mask(scores: mx.array, mask: Optional[mx.array]) -> mx.array:
    if mask is None:
        return scores
    if isinstance(mask, str):
        qL, kL = scores.shape[-2:]
        q_indices = mx.arange(kL - qL, kL)
        k_indices = mx.arange(kL)
        mask = q_indices[:, None] >= k_indices[None]
    elif mask.ndim == 3:
        if scores.ndim == 5:
            mask = mask[:, None, None, :, :]
        else:
            mask = mask[:, None, :, :]
    if mask.dtype == mx.bool_:
        return mx.where(
            mask,
            scores,
            mx.array(mx.finfo(scores.dtype).min, scores.dtype),
        )
    return scores + mask


def _bidirectional_mask(
    length: int,
    dtype: mx.Dtype,
    window_size: Optional[int] = None,
) -> Optional[mx.array]:
    if length <= 1 and window_size is None:
        return None
    if window_size is None:
        return None

    inds = mx.arange(length)
    qinds = inds[:, None]
    kinds = inds[None]
    left = qinds - kinds
    right = kinds - qinds
    left_window = (window_size + 1) // 2
    right_window = window_size // 2 + 1
    mask = ((left >= 0) & (left < left_window)) | (
        (right > 0) & (right < right_window)
    )
    return mask.astype(dtype)


def _cross_mask(
    decoder_len: int,
    encoder_attention_mask: Optional[mx.array],
    dtype: mx.Dtype,
) -> mx.array:
    if encoder_attention_mask is None:
        return mx.ones((decoder_len, 0), dtype=dtype)
    if encoder_attention_mask.ndim == 2:
        return mx.broadcast_to(
            encoder_attention_mask[:, None, :],
            (
                encoder_attention_mask.shape[0],
                decoder_len,
                encoder_attention_mask.shape[1],
            ),
        )
    return encoder_attention_mask


def _concat_masks(self_mask, cross_mask, dtype):
    if self_mask is None:
        if cross_mask.ndim == 2:
            self_mask = mx.ones((cross_mask.shape[0], 0), dtype=dtype)
        else:
            self_mask = mx.ones(
                (cross_mask.shape[0], cross_mask.shape[1], 0),
                dtype=dtype,
            )
    if cross_mask.ndim == 2:
        return mx.concatenate([self_mask, cross_mask], axis=-1)
    if self_mask.ndim == 2:
        self_mask = mx.expand_dims(self_mask, 0)
    return mx.concatenate([self_mask, cross_mask], axis=-1)


def _scaled_dot_product_attention_with_softcap(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[mx.array],
    softcap: float,
) -> mx.array:
    B, n_heads, L, head_dim = queries.shape
    n_kv_heads = keys.shape[1]
    repeats = n_heads // n_kv_heads

    queries = queries * scale
    if repeats > 1:
        queries = queries.reshape(B, n_kv_heads, repeats, L, head_dim)
        keys = mx.expand_dims(keys, 2)
        values = mx.expand_dims(values, 2)

    scores = queries @ keys.swapaxes(-1, -2)
    scores = mx.tanh(scores / softcap) * softcap
    scores = _attention_mask(scores, mask)
    scores = mx.softmax(scores, precise=True, axis=-1)
    output = scores @ values
    if repeats > 1:
        output = output.reshape(B, n_heads, L, head_dim)
    return output


class MLP(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class SelfAttention(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.layer_type = args.layer_types[layer_idx]
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim
        self.scale = args.query_pre_attn_scalar**-0.5
        self.attn_logit_softcapping = args.attn_logit_softcapping

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.n_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, args.rms_norm_eps)

        rope_params = args.rope_parameters[self.layer_type]
        self.rope = initialize_rope(
            dims=self.head_dim,
            traditional=False,
            base=rope_params.get("rope_theta", 10000.0),
            scaling_config=rope_params,
            max_position_embeddings=args.max_position_embeddings,
        )

    def _project(self, x: mx.array):
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        return self.q_norm(queries), self.k_norm(keys), values

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries, keys, values = self._project(x)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if self.attn_logit_softcapping is not None:
            output = _scaled_dot_product_attention_with_softcap(
                queries,
                keys,
                values,
                self.scale,
                mask,
                self.attn_logit_softcapping,
            )
        else:
            output = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=self.scale,
                mask=mask,
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MergedAttention(SelfAttention):
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        cache: Optional[CacheList],
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        B, L, _ = x.shape
        queries, keys, values = self._project(x)

        self_cache = cache[0] if cache is not None else None
        cross_cache = cache[1] if cache is not None else None

        if self_cache is not None:
            queries = self.rope(queries, offset=self_cache.offset)
            keys = self.rope(keys, offset=self_cache.offset)
            keys, values = self_cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if cross_cache is not None and not cross_cache.empty():
            cross_keys, cross_values = cross_cache.state
        else:
            cross_keys = self.k_proj(encoder_hidden_states)
            cross_values = self.v_proj(encoder_hidden_states)
            cross_keys = cross_keys.reshape(B, -1, self.n_kv_heads, self.head_dim)
            cross_values = cross_values.reshape(B, -1, self.n_kv_heads, self.head_dim)
            cross_keys = self.k_norm(cross_keys.transpose(0, 2, 1, 3))
            cross_values = cross_values.transpose(0, 2, 1, 3)
            if cross_cache is not None:
                cross_keys, cross_values = cross_cache.update_and_fetch(
                    cross_keys,
                    cross_values,
                )

        keys = mx.concatenate([keys, cross_keys], axis=2)
        values = mx.concatenate([values, cross_values], axis=2)

        if self.attn_logit_softcapping is not None:
            output = _scaled_dot_product_attention_with_softcap(
                queries,
                keys,
                values,
                self.scale,
                mask,
                self.attn_logit_softcapping,
            )
        else:
            output = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=None,
                scale=self.scale,
                mask=mask,
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class EncoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.self_attn = SelfAttention(args, layer_idx)
        self.pre_self_attn_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_self_attn_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.mlp = MLP(args)

    def __call__(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        r = self.self_attn(self.pre_self_attn_layernorm(x), mask, None)
        h = x + self.post_self_attn_layernorm(r)
        r = self.mlp(self.pre_feedforward_layernorm(h))
        return h + self.post_feedforward_layernorm(r)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.self_attn = MergedAttention(args, layer_idx)
        self.pre_self_attn_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_self_attn_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.mlp = MLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        cache: Optional[CacheList],
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        r = self.self_attn(
            self.pre_self_attn_layernorm(x),
            mask,
            cache,
            encoder_hidden_states,
        )
        h = x + self.post_self_attn_layernorm(r)
        r = self.mlp(self.pre_feedforward_layernorm(h))
        return h + self.post_feedforward_layernorm(r)


class TextEncoder(nn.Module):
    def __init__(self, args: TextArgs, eoi_token_index: int):
        super().__init__()
        self.args = args
        self.embed_tokens = ScaledEmbedding(args, eoi_token_index)
        self.layers = [
            EncoderLayer(args, layer_idx) for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        input_embeddings: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        full_mask = attention_mask
        if attention_mask is not None and attention_mask.ndim == 2:
            full_mask = attention_mask[:, None, :]
        sliding_mask = _bidirectional_mask(
            h.shape[1],
            mx.bool_,
            window_size=self.args.sliding_window,
        )
        if full_mask is not None and sliding_mask is not None:
            sliding_mask = sliding_mask & full_mask

        for layer, layer_type in zip(self.layers, self.args.layer_types):
            mask = full_mask if layer_type == "full_attention" else sliding_mask
            h = layer(h, mask)
        return self.norm(h)


class Decoder(nn.Module):
    def __init__(self, args: TextArgs, eoi_token_index: int):
        super().__init__()
        self.args = args
        self.embed_tokens = ScaledEmbedding(args, eoi_token_index)
        self.layers = [
            DecoderLayer(args, layer_idx) for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def _mask(
        self,
        h: mx.array,
        cache: list[CacheList],
        encoder_hidden_states: mx.array,
        encoder_attention_mask: Optional[mx.array],
        layer_type: str,
        layer_cache: Optional[CacheList],
    ) -> mx.array:
        self_cache = layer_cache[0] if layer_cache is not None else None
        if layer_type == "sliding_attention":
            self_mask = (
                self_cache.make_mask(
                    h.shape[1],
                    window_size=self.args.sliding_window,
                    return_array=True,
                )
                if self_cache is not None
                else create_causal_mask(h.shape[1], window_size=self.args.sliding_window)
            )
        else:
            self_mask = (
                self_cache.make_mask(h.shape[1], window_size=None, return_array=True)
                if self_cache is not None
                else create_causal_mask(h.shape[1])
            )
        if self_mask is None and h.shape[1] == 1 and encoder_attention_mask is None:
            return None
        if self_mask is None and self_cache is not None:
            self_len = self_cache.size() + h.shape[1]
            self_mask = mx.ones((h.shape[1], self_len), dtype=mx.bool_)
        cross = mx.ones(
            (h.shape[0], h.shape[1], encoder_hidden_states.shape[1]),
            dtype=mx.bool_,
        )
        if encoder_attention_mask is not None:
            cross = cross & _cross_mask(h.shape[1], encoder_attention_mask, mx.bool_)
        return _concat_masks(self_mask, cross, mx.bool_)

    def __call__(
        self,
        inputs: mx.array,
        encoder_hidden_states: mx.array,
        cache: Optional[list[CacheList]] = None,
        input_embeddings: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c, layer_type in zip(self.layers, cache, self.args.layer_types):
            mask = self._mask(
                h,
                cache,
                encoder_hidden_states,
                encoder_attention_mask,
                layer_type,
                c,
            )
            h = layer(h, mask, c, encoder_hidden_states)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.is_encoder_decoder = True
        self.bos_token_id = args.bos_token_id
        self.encoder_args = TextArgs.from_dict(args.encoder_text_config)
        self.decoder_args = TextArgs.from_dict(args.decoder_config)
        self.encoder = TextEncoder(self.encoder_args, args.eoi_token_index)
        self.decoder = Decoder(self.decoder_args, args.eoi_token_index)
        self.final_logit_softcapping = self.decoder_args.final_logit_softcapping

    def encode(
        self,
        inputs: mx.array,
        input_embeddings: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        return self.encoder(inputs, input_embeddings, attention_mask)

    def decode(
        self,
        inputs: mx.array,
        encoder_hidden_states: mx.array,
        cache: Optional[list[CacheList]] = None,
        input_embeddings: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.decoder(
            inputs,
            encoder_hidden_states,
            cache=cache,
            input_embeddings=input_embeddings,
            encoder_attention_mask=encoder_attention_mask,
        )
        out = self.encoder.embed_tokens.as_linear(out)
        if self.final_logit_softcapping is not None:
            out = mx.tanh(out / self.final_logit_softcapping)
            out = out * self.final_logit_softcapping
        return out

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[list[CacheList]] = None,
        input_embeddings: Optional[mx.array] = None,
        encoder_outputs: Optional[mx.array] = None,
        encoder_attention_mask: Optional[mx.array] = None,
        decoder_inputs: Optional[mx.array] = None,
        decoder_input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if encoder_outputs is None:
            encoder_outputs = self.encode(inputs, input_embeddings, encoder_attention_mask)
        if decoder_inputs is None:
            decoder_inputs = mx.full(
                (inputs.shape[0], 1),
                self.bos_token_id,
                dtype=inputs.dtype,
            )
        return self.decode(
            decoder_inputs,
            encoder_outputs,
            cache=cache,
            input_embeddings=decoder_input_embeddings,
            encoder_attention_mask=encoder_attention_mask,
        )

    def make_cache(self):
        caches = []
        for layer_type in self.decoder_args.layer_types:
            if layer_type == "sliding_attention":
                self_cache = RotatingKVCache(max_size=self.decoder_args.sliding_window)
            else:
                self_cache = KVCache()
            caches.append(CacheList(self_cache, KVCache()))
        return caches

    @property
    def layers(self):
        return self.decoder.layers

    @property
    def quant_predicate(self):
        def predicate(_, module):
            return not isinstance(module, ScaledEmbedding)

        return predicate

    def sanitize(self, weights):
        sanitized = {}
        encoder_embed_weight = None
        encoder_eoi_embedding = None

        for key, value in weights.items():
            key = key.removeprefix("model.")
            if key.startswith(
                (
                    "encoder.vision_tower.",
                    "encoder.multi_modal_projector.",
                )
            ):
                continue
            if "rotary_emb" in key:
                continue
            if key == "lm_head.out_proj.weight":
                continue

            key = key.replace("encoder.text_model.", "encoder.")
            if key.startswith("decoder."):
                key = key
            if key == "encoder.embed_tokens.weight":
                encoder_embed_weight = value
            elif key == "encoder.embed_tokens.eoi_embedding":
                encoder_eoi_embedding = value
            sanitized[key] = value

        if (
            "decoder.embed_tokens.weight" not in sanitized
            and encoder_embed_weight is not None
        ):
            sanitized["decoder.embed_tokens.weight"] = encoder_embed_weight
        if (
            "decoder.embed_tokens.eoi_embedding" not in sanitized
            and encoder_eoi_embedding is not None
        ):
            sanitized["decoder.embed_tokens.eoi_embedding"] = encoder_eoi_embedding

        return sanitized
