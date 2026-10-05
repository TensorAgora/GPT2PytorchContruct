"""python -m examples.execution_trace [--include-values [N]] [--out artifacts/execution_trace.json]"""
from gpt2_pytorch.data import FIXED_INPUT_IDS
from gpt2_pytorch.tracing import build_trace, write_trace
from gpt2_pytorch.utils.cli import common_parser, load_model

parser = common_parser(__doc__)
parser.add_argument("--out", default="artifacts/execution_trace.json")
parser.add_argument("--include-values", nargs="?", type=int, const=1024, default=0, metavar="MAX_NUMEL",
                    help="also serialize values of tensors with at most MAX_NUMEL elements (default 1024)")
args = parser.parse_args()

model, _, _ = load_model(args)
trace = build_trace(model, FIXED_INPUT_IDS.to(next(model.parameters()).device), max_values=args.include_values)
path = write_trace(trace, args.out)

print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB): format={trace['format']} v{trace['version']}")
print(f"{len(trace['parameters'])} parameters, {len(trace['buffers'])} buffers, {len(trace['events'])} events")
for e in trace["events"][:12]:
    print(f"{e['index']:>3} {e['module']:<62}{e['module_type']:<20}{e['input_shapes']} -> {e['output_shapes']}")
print("...")
attn = next(e for e in trace["events"] if e['module'].endswith("blocks.0.attention"))
print("attention internals:", [(t["name"], t["shape"]) for t in attn["internals"]])
print("outputs:", [(t["name"], t["shape"]) for t in trace["outputs"]])
