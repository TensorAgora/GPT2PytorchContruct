"""TensorRT engine (.engine) from the static ONNX export. Needs an NVIDIA GPU with CUDA and the `tensorrt` package
(Linux/Windows). Engines are specific to the TensorRT version and GPU that built them, and their layout is undocumented.

Without TensorRT this records a skip and the equivalent command line (also run it there to get the layer info JSON):
    trtexec --onnx=<name>.onnx --saveEngine=<name>.engine --exportLayerInfo=<name>.layers.json --profilingVerbosity=detailed
"""
import torch

from .common import Manifest, compare, load_source, parse_args, reference_logits

args = parse_args(__doc__)
out, m, name = args.out / "tensorrt", Manifest(args, "tensorrt"), args.model
onnx_path = args.out / "onnx" / f"{name}.onnx"
command = (f"trtexec --onnx={onnx_path} --saveEngine={out / f'{name}.engine'} "
           f"--exportLayerInfo={out / f'{name}.layers.json'} --profilingVerbosity=detailed")

try:
    import tensorrt as trt
except ImportError:
    trt = None

if trt is None or not torch.cuda.is_available():
    m.skip("tensorrt", ".engine from ONNX", f"needs TensorRT + an NVIDIA CUDA GPU (tensorrt importable: {trt is not None}, "
           f"CUDA: {torch.cuda.is_available()}); on such a machine run this script or: {command}")
elif not onnx_path.exists():
    m.skip("tensorrt", ".engine from ONNX", f"{onnx_path} missing: run onnx_format first")
else:
    out.mkdir(exist_ok=True)
    model, ids = load_source(args.model)
    ref = reference_logits(model, ids)

    def build():  # ponytail: untested here (no NVIDIA GPU on the machine that wrote it)
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        engine_bytes = builder.build_serialized_network(network, config)
        p = out / f"{name}.engine"
        p.write_bytes(engine_bytes)
        engine = trt.Runtime(logger).deserialize_cuda_engine(p.read_bytes())
        (out / f"{name}.layers.json").write_text(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
        context = engine.create_execution_context()
        x = ids.to(torch.int64).cuda()
        y = torch.empty(tuple(engine.get_tensor_shape("logits")), dtype=torch.float32, device="cuda")
        context.set_tensor_address("input_ids", x.data_ptr())
        context.set_tensor_address("logits", y.data_ptr())
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return [p, out / f"{name}.layers.json"], compare(ref, y.cpu().numpy())

    m.variant("tensorrt", ".engine from ONNX", "opaque engine (+ layer info JSON)", build, f"equivalent: {command}", tol=1e-2)
m.write()
