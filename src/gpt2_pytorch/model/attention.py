import math

import torch
from torch import nn

from ..config import GPT2Config

# debug-dict key -> Tensormorph trace tensor name (block.N.attention.<name>)
TRACE_NAMES = {
    "qkv": "qkv",
    "query": "q",
    "key": "k",
    "value": "v",
    "attention_scores": "scores",
    "causal_mask": "mask",
    "attention_probs": "probs",
    "context": "context",
}


def flatten_attention_debug(prefix: str, debug: dict) -> dict:
    return {f"{prefix}.{TRACE_NAMES[name]}": tensor for name, tensor in debug.items()}


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with every intermediate tensor spelled out.

    B batch, T sequence, C hidden, H heads, D head_dim (C = H * D).
    Deliberately avoids nn.MultiheadAttention and scaled_dot_product_attention.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.qkv_projection = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

        # True = position may be attended to (lower triangle). Non-persistent: not part of state_dict.
        n = config.max_position_embeddings
        self.register_buffer("causal_mask", torch.tril(torch.ones(n, n, dtype=torch.bool)), persistent=False)

        # Set by ActivationRecorder to receive the debug dict without changing the call site.
        self.debug_sink = None

    def _split_heads(self, x):
        # [B, T, C] -> [B, T, H, D] -> [B, H, T, D]
        x = x.view(x.size(0), x.size(1), self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        # [B, H, T, D] -> [B, T, H, D] -> [B, T, C]
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(x.size(0), x.size(1), self.hidden_size)

    def forward(self, hidden_states, return_debug=False):
        # [B, T, C] -> [B, T, 3C]
        qkv = self.qkv_projection(hidden_states)

        # three tensors, each [B, T, C]
        query, key, value = qkv.chunk(3, dim=-1)

        # [B, T, C] -> [B, H, T, D]
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # [B, H, T, D] @ [B, H, D, T] -> [B, H, T, T]
        query_key_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = query_key_scores / math.sqrt(self.head_dim)

        # [T, T] bool, True where attention is allowed
        seq_len = attention_scores.size(-1)
        causal_mask = torch.narrow(torch.narrow(self.causal_mask, 0, 0, seq_len), 1, 0, seq_len)  # == mask[:T, :T]

        # -inf (not finfo.min) keeps this dtype-agnostic and traceable; softmax gives exact 0 for future positions.
        masked_scores = attention_scores.masked_fill(~causal_mask, float("-inf"))

        # softmax over the key axis: each query row sums to 1.  [B, H, T, T]
        attention_probs = torch.softmax(masked_scores, dim=-1)
        attention_probs_dropped = self.attention_dropout(attention_probs)

        # [B, H, T, T] @ [B, H, T, D] -> [B, H, T, D]
        context = torch.matmul(attention_probs_dropped, value)

        # [B, H, T, D] -> [B, T, C]
        merged_context = self._merge_heads(context)

        # [B, T, C] -> [B, T, C]
        output = self.output_projection(merged_context)
        output = self.residual_dropout(output)

        if return_debug or self.debug_sink is not None:
            debug = {
                "qkv": qkv,
                "query": query,
                "key": key,
                "value": value,
                "attention_scores": attention_scores,  # scaled, before masking
                "causal_mask": causal_mask,
                "attention_probs": attention_probs,  # after softmax, before dropout
                "context": context,  # [B, H, T, D], before head merge
            }
            if self.debug_sink is not None:
                self.debug_sink(debug)
            if return_debug:
                return output, debug
        return output
