"""Core ML via coremltools from a torch.jit.trace of the model (static [2, 8] int32 input):
mlprogram `.mlpackage` (FP32 and FP16 compute) and the older neuralnetwork `.mlmodel`. Each is run with Core ML (macOS)."""
import shutil

import coremltools as ct
import numpy as np
import torch

from .common import Manifest, compare, load_source, parse_args, reference_logits

args = parse_args(__doc__)
model, ids = load_source(args.model)
ref = reference_logits(model, ids)
out, m, name = args.out / "coreml", Manifest(args, "coreml"), args.model
out.mkdir(exist_ok=True)
with torch.no_grad():
    traced = torch.jit.trace(model, (ids,), strict=False)
INPUTS = [ct.TensorType(name="input_ids", shape=tuple(ids.shape), dtype=np.int32)]


def predict(mlmodel):
    result = mlmodel.predict({"input_ids": ids.numpy().astype(np.int32)})
    return compare(ref, next(iter(result.values())))


def convert(path, **kwargs):
    def fn():
        if path.exists():
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        mlmodel = ct.convert(traced, inputs=INPUTS, outputs=[ct.TensorType(name="logits")], **kwargs)
        mlmodel.save(str(path))
        return [path], predict(ct.models.MLModel(str(path)))
    return fn


m.variant("coreml", "mlprogram FP32 .mlpackage", "MIL program", convert(out / f"{name}.mlpackage", convert_to="mlprogram",
          compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.macOS13),
          "directory: Manifest.json + Data/com.apple.CoreML/{model.mlmodel, weights/weight.bin}")
m.variant("coreml", "mlprogram FP16 .mlpackage", "MIL program", convert(out / f"{name}.fp16.mlpackage", convert_to="mlprogram",
          compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS13), "weights and compute in fp16", tol=5e-2)
m.variant("coreml", "neuralnetwork .mlmodel", "NeuralNetwork layers", convert(out / f"{name}.mlmodel", convert_to="neuralnetwork"),
          "single protobuf file, weights inline")
m.write()
