import torch
from torch.utils.data import DataLoader, Dataset

from .fixed_samples import SAMPLES


class FixedTokenDataset(Dataset):
    """Rows of a fixed [N, T] id tensor. labels == input_ids; the causal LM loss does the shift."""

    def __init__(self, input_ids: torch.Tensor | str = "sample_batch"):
        self.input_ids = SAMPLES[input_ids] if isinstance(input_ids, str) else input_ids

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, index):
        row = self.input_ids[index]
        return {"input_ids": row, "labels": row.clone()}


def fixed_dataloader(sample: str = "sample_batch", batch_size: int = 2) -> DataLoader:
    return DataLoader(FixedTokenDataset(sample), batch_size=batch_size, shuffle=False)
