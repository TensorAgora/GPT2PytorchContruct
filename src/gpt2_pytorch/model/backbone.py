from torch import nn

from ..config import GPT2Config
from .block import TransformerBlock
from .embeddings import GPT2Embeddings


class GPT2Model(nn.Module):
    """input_ids [B, T] -> final hidden states [B, T, C]."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.embeddings = GPT2Embeddings(config)  # token + position embedding + dropout
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    @property
    def token_embedding(self):
        return self.embeddings.token_embedding

    @property
    def position_embedding(self):
        return self.embeddings.position_embedding

    @property
    def embedding_dropout(self):
        return self.embeddings.dropout

    def forward(self, input_ids, position_ids=None, return_debug=False):
        debug = {"input.input_ids": input_ids} if return_debug else None

        # [B, T] -> [B, T, C]
        if return_debug:
            hidden_states, embedding_debug = self.embeddings(input_ids, position_ids, return_debug=True)
            debug.update({f"embedding.{name}": tensor for name, tensor in embedding_debug.items()})
        else:
            hidden_states = self.embeddings(input_ids, position_ids)

        for index, block in enumerate(self.blocks):
            # [B, T, C] -> [B, T, C]
            if return_debug:
                hidden_states, block_debug = block(hidden_states, return_debug=True)
                debug.update({f"block.{index}.{name}": tensor for name, tensor in block_debug.items()})
            else:
                hidden_states = block(hidden_states)

        # [B, T, C] -> [B, T, C]
        final_hidden = self.final_layer_norm(hidden_states)

        if return_debug:
            debug["model.final_hidden"] = final_hidden
            return final_hidden, debug
        return final_hidden
