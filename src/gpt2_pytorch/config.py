from dataclasses import dataclass


@dataclass(frozen=True)
class GPT2Config:
    """DistilGPT2 hyper-parameters. Defaults are the values verified against the checkpoint."""

    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 12
    intermediate_size: int = 3072
    layer_norm_epsilon: float = 1e-5
    embedding_dropout: float = 0.1
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1

    def __post_init__(self):
        if self.hidden_size % self.num_heads:
            raise ValueError(f"hidden_size {self.hidden_size} not divisible by num_heads {self.num_heads}")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads
