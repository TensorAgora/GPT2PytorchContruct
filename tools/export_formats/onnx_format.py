"""ONNX: dynamo exporter (static and dynamic batch/sequence), legacy TorchScript exporter, external-data layout,
dynamic int8 quantization. Each model is checked with onnx.checker and run in onnxruntime against PyTorch logits."""
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from .common import Manifest, compare, load_source, parse_args, reference_logits

args = parse_args(__doc__)
model, ids = load_source(args.model)
ref = reference_logits(model, ids)
out, m, name = args.out / "onnx", Manifest(args, "onnx"), args.model
out.mkdir(exist_ok=True)


def run(path, x=ids):
    onnx.checker.check_model(str(path))  # path form: also validates external data
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {session.get_inputs()[0].name: x.numpy()}
    return session.run(None, feed)[0]


def dynamo_static():
    p = out / f"{name}.onnx"
    torch.onnx.export(model, (ids,), p, dynamo=True, external_data=False, input_names=["input_ids"], output_names=["logits"])
    return [p], compare(ref, run(p))


def dynamo_dynamic():
    p = out / f"{name}.dynamic.onnx"
    dims = {"input_ids": {0: torch.export.Dim("batch", max=64), 1: torch.export.Dim("sequence", min=2, max=model.config.max_position_embeddings)}}
    torch.onnx.export(model, (ids,), p, dynamo=True, dynamic_shapes=dims, input_names=["input_ids"], output_names=["logits"])
    one = ids[:1, :5]  # a different shape than the export example
    data = p.with_name(p.name + ".data")
    return [p, *([data] if data.exists() else [])], compare(ref, run(p)) | {"other_shape_max_abs_diff": compare(reference_logits(model, one), run(p, one))["max_abs_diff"]}


def legacy_exporter():
    p = out / f"{name}.legacy_opset17.onnx"
    torch.onnx.export(model, (ids,), p, dynamo=False, opset_version=17, input_names=["input_ids"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}})
    return [p], compare(ref, run(p))


def external_data():
    p, data = out / f"{name}.external.onnx", f"{name}.external.bin"
    onnx.save_model(onnx.load(str(out / f"{name}.onnx")), str(p), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=data, size_threshold=1024)  # small shape constants stay inline
    return [p, out / data], compare(ref, run(p))


def int8_dynamic():
    p = out / f"{name}.int8_dynamic.onnx"
    quantize_dynamic(str(out / f"{name}.legacy_opset17.onnx"), str(p), weight_type=QuantType.QInt8, per_channel=True)
    return [p], compare(ref, run(p))


m.variant("onnx", "dynamo exporter, static [2,8]", "ONNX graph", dynamo_static, "weights inline in the .onnx")
m.variant("onnx", "dynamo exporter, dynamic axes", "ONNX graph", dynamo_dynamic, "symbolic batch / sequence; weights in .onnx.data (exporter default)")
m.variant("onnx", "legacy exporter, opset 17", "ONNX graph", legacy_exporter, "TorchScript-based exporter (dynamo=False): Gemm/MatMul patterns differ")
m.variant("onnx", "external data (.onnx + .bin)", "ONNX graph", external_data, "all initializers in one sibling .bin file")
m.variant("onnx", "dynamic int8 (MatMulInteger)", "ONNX graph", int8_dynamic, "onnxruntime quantize_dynamic on the opset-17 model: per-channel int8 weights; activations quantized per tensor at run time, "
          "so lossy by design (DistilGPT2: ~11% of max |logit|)", tol=0.25)
m.write()
