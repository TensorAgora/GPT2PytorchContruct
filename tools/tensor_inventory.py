"""python -m tools.tensor_inventory [--no-export]"""
import argparse

from gpt2_pytorch.utils import build_inventory, write_inventory
from gpt2_pytorch.utils.cli import common_parser, load_model

parser = common_parser(__doc__)
parser.add_argument("--no-export", action="store_true")
parser.add_argument("--out-dir", default="artifacts")
args = parser.parse_args()

model, _, _ = load_model(args)
rows = build_inventory(model)
print(f"{'name':<62}{'role':<10}{'shape':<16}{'dtype':<9}{'numel':>11}{'bytes':>12}  shared")
for r in rows:
    shape = str(r["shape"]).replace(" ", "")
    print(f"{r['name']:<62}{r['role']:<10}{shape:<16}{r['dtype']:<9}{r['numel']:>11,}{r['bytes']:>12,}  {r['shared']}")
if not args.no_export:
    write_inventory(rows, f"{args.out_dir}/tensor_inventory.json", f"{args.out_dir}/tensor_inventory.csv")
    print(f"\nwrote {args.out_dir}/tensor_inventory.json and .csv ({len(rows)} rows)")
