import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F

from transformers.models.phi3.modeling_phi3 import (
    Phi3Attention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)
import types

__all__ = ["enable_phi3_pos_shift_attention"]


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    # cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    # sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def phi3_pos_shift_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    # assert False


    qkv = self.qkv_proj(hidden_states)
    query_pos = self.num_heads * self.head_dim
    query_states = qkv[..., :query_pos]
    key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
    value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None: 
        # kv_seq_len += past_key_value[0].shape[-2]
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    cos, sin = self.rotary_emb(value_states, position_ids)
    ### Shift Pos: query pos is min(cache_size, idx)
    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
    ###

    if past_key_value is not None:
        # reuse k, v, self_attention
        # key_states = torch.cat([past_key_value[0], key_states], dim=2)
        # value_states = torch.cat([past_key_value[1], value_states], dim=2)
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # past_key_value = (key_states, value_states) if use_cache else None

    ### Shift Pos: key pos is the pos in cache
    key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
    cos, sin = self.rotary_emb(key_states, key_position_ids)
    key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
    ###

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    # try:
    if attention_mask is not None:
        # if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
        #     raise ValueError(
        #         f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
        #     )
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        # attn_weights = attn_weights + attention_mask
        attn_weights = attn_weights + causal_mask
    # except:
    #     breakpoint()

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def enable_phi3_pos_shift_attention(model):
    for name, module in reversed(model._modules.items()): #从后往前遍历模型的所有层
        if len(list(module.children())) > 0: #如果当前层还有子层
            enable_phi3_pos_shift_attention( #递归调用
                module,
            )

        if isinstance(module, Phi3Attention): #如果当前层是LlamaAttention
            model._modules[name].forward = types.MethodType( #将forward函数替换为phi3_pos_shift_attention_forward
                phi3_pos_shift_attention_forward, model._modules[name]
            )