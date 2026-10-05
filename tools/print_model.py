"""python -m tools.print_model [--full]   (random init is enough: only structure is printed)"""
import argparse

from gpt2_pytorch import DistilGPT2LMHeadModel
from gpt2_pytorch.utils import format_breakdown, module_tree, parameter_breakdown

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--full", action="store_true", help="print all 6 blocks")
args = parser.parse_args()

model = DistilGPT2LMHeadModel()
print(module_tree(model, collapse=not args.full))
print()
print(format_breakdown(parameter_breakdown(model)))
