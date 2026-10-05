import torch

from gpt2_pytorch.data import FIXED_INPUT_IDS, SAMPLES, FixedTokenDataset, fixed_dataloader


def test_samples_are_fixed_and_valid():
    assert FIXED_INPUT_IDS.tolist() == [[15496, 11, 616, 1438, 318, 1332, 13, 198], [1212, 318, 257, 1332, 6827, 284, 1332, 13]]
    assert set(SAMPLES) == {"sample_single", "sample_batch", "sample_repeated_tokens", "sample_short_sequence", "sample_attention_debug"}
    for ids in SAMPLES.values():
        assert ids.dtype == torch.long and ids.dim() == 2 and 0 <= ids.min() and ids.max() < 50257


def test_dataloader_is_deterministic_and_unshuffled():
    runs = [[b["input_ids"] for b in fixed_dataloader(batch_size=1)] for _ in range(2)]
    assert all(torch.equal(a, b) for a, b in zip(*runs))
    assert torch.equal(runs[0][0][0], FIXED_INPUT_IDS[0]) and torch.equal(runs[0][1][0], FIXED_INPUT_IDS[1])


def test_batch_structure():
    batch = next(iter(fixed_dataloader("sample_batch", batch_size=2)))
    assert batch["input_ids"].shape == batch["labels"].shape == (2, 8)
    assert torch.equal(batch["input_ids"], batch["labels"]) and batch["labels"] is not batch["input_ids"]
    assert len(FixedTokenDataset("sample_short_sequence")) == 2
