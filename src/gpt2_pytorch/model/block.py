from torch import nn

from ..config import GPT2Config
from .attention import CausalSelfAttention, flatten_attention_debug
from .mlp import GPT2MLP


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block: x + Attn(LN1(x)), then x + MLP(LN2(x))."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        eps = config.layer_norm_epsilon
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=eps)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=eps)
        self.mlp = GPT2MLP(config)

    def forward(self, hidden_states, return_debug=False):
        block_input = hidden_states

        # ---- attention sub-layer ----
        residual = hidden_states
        normalized = self.ln_1(hidden_states)
        if return_debug:
            attention_output, attention_debug = self.attention(normalized, return_debug=True)
        else:
            attention_output = self.attention(normalized)
        hidden_states = residual + attention_output
        after_attention = hidden_states

        # ---- MLP sub-layer ----
        residual = hidden_states
        normalized_2 = self.ln_2(hidden_states)
        if return_debug:
            mlp_output, mlp_debug = self.mlp(normalized_2, return_debug=True)
        else:
            mlp_output = self.mlp(normalized_2)
        hidden_states = residual + mlp_output

        if return_debug:
            debug = {"input": block_input, "ln_1.output": normalized}
            debug.update(flatten_attention_debug("attention", attention_debug))
            debug["attention.output"] = attention_output
            debug["attention_residual.output"] = after_attention
            debug["ln_2.output"] = normalized_2
            debug.update({f"mlp.{name}": tensor for name, tensor in mlp_debug.items()})
            debug["mlp.output"] = mlp_output
            debug["output"] = hidden_states
            return hidden_states, debug
        return hidden_states
