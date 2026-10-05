import math

from torch import nn
import torch

from ..config import GPT2Config


class GPT2GELU(nn.Module):
    """GPT-2's 'gelu_new' (tanh approximation), written out. Not the exact erf GELU of F.gelu(x)."""

    def forward(self, x):
        cubic = x * x * x
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * cubic)
        return 0.5 * x * (1.0 + torch.tanh(inner))


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.fc = nn.Linear(config.hidden_size, config.intermediate_size)  # C -> 4C (HF: c_fc)
        self.activation = GPT2GELU()
        self.projection = nn.Linear(config.intermediate_size, config.hidden_size)  # 4C -> C (HF: c_proj)
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(self, hidden_states, return_debug=False):
        # [B, T, C] -> [B, T, 4C]
        fc_output = self.fc(hidden_states)
        # [B, T, 4C] -> [B, T, 4C]
        activation_output = self.activation(fc_output)
        # [B, T, 4C] -> [B, T, C]
        output = self.projection(activation_output)
        output = self.dropout(output)

        if return_debug:
            return output, {"fc.output": fc_output, "activation.output": activation_output}
        return output
