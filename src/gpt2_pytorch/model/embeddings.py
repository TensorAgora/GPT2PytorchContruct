import torch
from torch import nn

from ..config import GPT2Config


class TokenEmbedding(nn.Embedding):
    """wte: token id -> vector. Weight [vocab_size, C]. Also the tied LM-head matrix."""


class PositionEmbedding(nn.Embedding):
    """wpe: absolute position -> vector. Weight [max_positions, C] (learned, not sinusoidal)."""


class GPT2Embeddings(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.token_embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.position_embedding = PositionEmbedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.embedding_dropout)
        # [N] = 0..N-1, sliced to [:T] in forward (no arange on a traced shape -> FX/export friendly)
        positions = torch.arange(config.max_position_embeddings)
        self.register_buffer("position_ids", positions, persistent=False)

    def forward(self, input_ids, position_ids=None, return_debug=False):
        # input_ids [B, T]
        seq_len = input_ids.size(1)
        torch._assert(seq_len <= self.position_ids.size(0), "sequence longer than max_position_embeddings")
        if position_ids is None:
            # [N] -> [T] -> [1, T] (broadcast over batch). narrow == [:T] but traceable by FX with a symbolic T.
            position_ids = torch.narrow(self.position_ids, 0, 0, seq_len).unsqueeze(0)

        # [B, T] -> [B, T, C]
        token_embeddings = self.token_embedding(input_ids)
        # [1, T] -> [1, T, C]
        position_embeddings = self.position_embedding(position_ids)

        # [B, T, C] + [1, T, C] -> [B, T, C]
        hidden_states = token_embeddings + position_embeddings
        hidden_states = self.dropout(hidden_states)

        if return_debug:
            debug = {
                "token.output": token_embeddings,
                "position.output": position_embeddings,
                "output": hidden_states,
            }
            return hidden_states, debug
        return hidden_states
