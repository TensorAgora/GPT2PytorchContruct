"""python -m tools.inspect_checkpoint [--checkpoint PATH] [--full]"""
import argparse

from gpt2_pytorch.checkpoint import DEFAULT_CHECKPOINT
from gpt2_pytorch.checkpoint.inspect import describe_checkpoint

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
parser.add_argument("--full", action="store_true", help="list every block, not just block 0")
args = parser.parse_args()
print(describe_checkpoint(args.checkpoint, full=args.full))
