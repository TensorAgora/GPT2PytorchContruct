"""python -m examples.dataloader_forward: Dataset -> DataLoader -> batch -> model -> logits."""
import torch

from gpt2_pytorch.data import FixedTokenDataset, fixed_dataloader
from gpt2_pytorch.tracing import fmt_shape
from gpt2_pytorch.utils.cli import common_parser, load_model

parser = common_parser(__doc__)
parser.add_argument("--sample", default="sample_batch")
parser.add_argument("--batch-size", type=int, default=2)
args = parser.parse_args()

model, _, _ = load_model(args)
device = next(model.parameters()).device
loader = fixed_dataloader(args.sample, args.batch_size)
print(f"dataset: {args.sample}, {len(loader.dataset)} rows, {len(loader)} batches, shuffle=False")
for step, batch in enumerate(loader):
    input_ids = batch["input_ids"].to(device)
    with torch.no_grad():
        logits = model(input_ids)
    print(f"batch {step}: input_ids {fmt_shape(input_ids.shape)} labels {fmt_shape(batch['labels'].shape)} -> logits {fmt_shape(logits.shape)}")
    print(f"  logits[:, -1, :3] = {logits[:, -1, :3].tolist()}")
