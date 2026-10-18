"""GGUF (llama.cpp): architecture "gpt2", tensor names and metadata keys from gguf-py's own tables, in f32 / f16 / q8_0 /
q4_0. Verification reads each file back with GGUFReader, dequantizes, rebuilds the PyTorch model and compares logits.

Layout follows llama.cpp's HF->GGUF converter for GPT-2: projection weights as [out, in] (HF Conv1D transposed), attn.bias
masks dropped. The head is tied, so no `output.weight` is written (llama.cpp falls back to token_embd). No tokenizer.*
keys: the files are complete for inspection, but llama.cpp will not load them for text generation."""
import re

import gguf
import numpy as np
import torch

from gpt2_pytorch import DistilGPT2LMHeadModel

from .common import Manifest, compare, load_source, parse_args, reference_logits

args = parse_args(__doc__)
model, ids = load_source(args.model)
ref = reference_logits(model, ids)
out, m, name, c = args.out / "gguf", Manifest(args, "gguf"), args.model, model.config
out.mkdir(exist_ok=True)

T = gguf.MODEL_TENSOR
ARCH = gguf.MODEL_ARCH.GPT2
TOP = {"transformer.embeddings.token_embedding": T.TOKEN_EMBD, "transformer.embeddings.position_embedding": T.POS_EMBD,
       "transformer.final_layer_norm": T.OUTPUT_NORM}
BLOCK = {"ln_1": T.ATTN_NORM, "attention.qkv_projection": T.ATTN_QKV, "attention.output_projection": T.ATTN_OUT,
         "ln_2": T.FFN_NORM, "mlp.fc": T.FFN_UP, "mlp.projection": T.FFN_DOWN}


def gguf_name(ours: str) -> str | None:
    module, kind = ours.rsplit(".", 1)
    if module in TOP:
        return f"{gguf.TENSOR_NAMES[TOP[module]]}.{kind}"
    if mm := re.fullmatch(r"transformer\.blocks\.(\d+)\.(.+)", module):
        return f"{gguf.TENSOR_NAMES[BLOCK[mm.group(2)]].format(bid=int(mm.group(1)))}.{kind}"
    return None  # lm_head.weight: tied to token_embd


TENSORS = {gguf_name(k): (k, v.numpy()) for k, v in model.state_dict().items() if gguf_name(k)}


def write(path, file_type, qtype):
    writer = gguf.GGUFWriter(str(path), gguf.MODEL_ARCH_NAMES[ARCH])
    writer.add_name(name)
    writer.add_description("Tensormorph export fixture from GPT2PytorchContruct (no tokenizer)")
    writer.add_context_length(c.max_position_embeddings)
    writer.add_embedding_length(c.hidden_size)
    writer.add_feed_forward_length(c.intermediate_size)
    writer.add_block_count(c.num_layers)
    writer.add_head_count(c.num_heads)
    writer.add_layer_norm_eps(c.layer_norm_epsilon)
    writer.add_file_type(file_type)
    for gname, (_, data) in TENSORS.items():
        if data.ndim == 1 or qtype == gguf.GGMLQuantizationType.F32:  # norms and biases stay f32, as llama.cpp does
            writer.add_tensor(gname, data.astype(np.float32))
        elif qtype == gguf.GGMLQuantizationType.F16:
            writer.add_tensor(gname, data.astype(np.float16))
        else:
            writer.add_tensor(gname, gguf.quants.quantize(data.astype(np.float32), qtype), raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def read_back(path):
    reader = gguf.GGUFReader(str(path))
    state = {}
    for t in reader.tensors:
        ours, source = TENSORS[t.name]
        values = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(source.shape)
        state[ours] = torch.from_numpy(values.copy())
    fresh = DistilGPT2LMHeadModel(c).eval()
    missing, unexpected = fresh.load_state_dict(state, strict=False)
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)  # tied alias, filled by the embedding
    arch = reader.fields["general.architecture"]
    return compare(ref, reference_logits(fresh, ids)) | {"tensors": len(reader.tensors), "architecture": bytes(arch.parts[arch.data[0]]).decode()}


def variant(file_type, qtype, suffix):
    def fn():
        p = out / f"{name}.{suffix}.gguf"
        write(p, file_type, qtype)
        return [p], read_back(p)
    return fn


Q = gguf.GGMLQuantizationType
F = gguf.LlamaFileType
m.variant("gguf", "f32", "none (arch + hyperparameters)", variant(F.ALL_F32, Q.F32, "f32"))
m.variant("gguf", "f16 (1-D tensors f32)", "none (arch + hyperparameters)", variant(F.MOSTLY_F16, Q.F16, "f16"), tol=1e-2)
m.variant("gguf", "q8_0 (1-D tensors f32)", "none (arch + hyperparameters)", variant(F.MOSTLY_Q8_0, Q.Q8_0, "q8_0"),
          "block-quantized: 32 values per 34-byte block", tol=0.1)
m.variant("gguf", "q4_0 (1-D tensors f32)", "none (arch + hyperparameters)", variant(F.MOSTLY_Q4_0, Q.Q4_0, "q4_0"),
          "block-quantized: 32 values per 18-byte block", tol=0.25)
m.write()
