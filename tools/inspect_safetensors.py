"""python -m tools.inspect_safetensors [PATH]   (reads only the JSON header, never the tensor data)"""
import argparse
import math

from gpt2_pytorch.checkpoint.safetensors_loader import read_safetensors_header

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("path", nargs="?", default="artifacts/model.safetensors")
args = parser.parse_args()

header = read_safetensors_header(args.path)
print(f"file: {args.path}\nheader bytes: {header.pop('__header_bytes__')}\nmetadata: {header.pop('__metadata__', {})}")
print(f"{'name':<56}{'shape':<18}{'dtype':<6}{'offsets (begin,end)':<26}{'numel':>12}{'bytes':>13}")
total = 0
for name, entry in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
    begin, end = entry["data_offsets"]
    numel = math.prod(entry["shape"])
    nbytes = numel * DTYPE_BYTES[entry["dtype"]]
    assert nbytes == end - begin, f"{name}: shape*dtype = {nbytes} but offsets span {end - begin}"
    total += nbytes
    print(f"{name:<56}{str(entry['shape']).replace(' ', ''):<18}{entry['dtype']:<6}{f'{begin},{end}':<26}{numel:>12,}{nbytes:>13,}")
print(f"\n{len(header)} tensors, {total:,} data bytes ({total / 2**20:.2f} MiB)")
