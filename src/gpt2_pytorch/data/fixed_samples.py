"""Hard-coded GPT-2 token ids. No tokenizer, no randomness, no network. All ids < 50257."""
import torch

# "Hello, my name is test.\n" / "This is a test sentence to test."
_BATCH = [
    [15496, 11, 616, 1438, 318, 1332, 13, 198],
    [1212, 318, 257, 1332, 6827, 284, 1332, 13],
]

SAMPLES: dict[str, torch.Tensor] = {
    "sample_single": torch.tensor(_BATCH[:1], dtype=torch.long),  # [1, 8]
    "sample_batch": torch.tensor(_BATCH, dtype=torch.long),  # [2, 8]
    "sample_repeated_tokens": torch.tensor([[1332] * 8, [262, 1332] * 4], dtype=torch.long),  # [2, 8]
    "sample_short_sequence": torch.tensor([[464, 2068, 7586, 21831], [18045, 625, 262, 16931]], dtype=torch.long),  # [2, 4]
    # "The quick brown fox jumps over the lazy dog" (9 tokens, 8 used) - distinct ids, easy to read attention rows
    "sample_attention_debug": torch.tensor([[464, 2068, 7586, 21831, 18045, 625, 262, 16931]], dtype=torch.long),  # [1, 8]
}

FIXED_INPUT_IDS = SAMPLES["sample_batch"]
