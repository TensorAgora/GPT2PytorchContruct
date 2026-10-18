"""OpenVINO IR: `<name>.xml` (graph) + `<name>.bin` (raw weights, no header: names, dtypes and offsets live in the .xml).
One more meaning of ".bin". Converted straight from the PyTorch model, run with the OpenVINO CPU runtime."""
import openvino as ov

from .common import Manifest, compare, load_source, parse_args, reference_logits


def main():
    args = parse_args(__doc__)
    model, ids = load_source(args.model)
    ref = reference_logits(model, ids)
    out, m, name = args.out / "openvino", Manifest(args, "openvino"), args.model
    out.mkdir(exist_ok=True)

    def ir(suffix, compress_to_fp16):
        def fn():
            xml = out / f"{name}{suffix}.xml"
            ov.save_model(ov.convert_model(model, example_input=ids), str(xml), compress_to_fp16=compress_to_fp16)
            compiled = ov.Core().compile_model(str(xml), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            return [xml, xml.with_suffix(".bin")], compare(ref, compiled(ids.numpy())[0])
        return fn

    m.variant("openvino", "IR fp32 (.xml + .bin)", "OpenVINO IR graph (.xml)", ir("", False))
    m.variant("openvino", "IR fp16-compressed weights", "OpenVINO IR graph (.xml)", ir(".fp16", True),
              "weights stored fp16 in the .bin, decompressed by Convert ops in the graph", tol=1e-2)
    m.write()


# guard: ov.convert_model starts worker processes that re-import this module; unguarded, each re-ran the export
if __name__ == "__main__":
    main()
