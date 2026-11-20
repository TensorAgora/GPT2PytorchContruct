"""Merge the Hugging Face model CSVs in csv/ and split them into category/format subfolders.

    uv run python tools/split_model_catalog.py            # writes csv/categories/
    uv run python tools/split_model_catalog.py --check    # also runs the self-check

Each category is a set of models whose computation graph is built the same way. Each folder holds
one CSV per weight format plus metadata_spec.json: the config.json / GGUF keys that feed the graph `dims`.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "csv"
OUT = SRC / "categories"

# Highest priority first: when several sources list the same id, the first non-empty value wins per column.
# hf_open_weight_models.csv is hf_llm_vlm + hf_timm (checked), so it adds nothing and is skipped.
SOURCES = [
    "deepseek_models_extended.csv", "deepseek_models.csv",
    "hf_llm_vlm_models.csv", "hf_timm_vision_models.csv",
    "huggingface_nlp_open_weight_models_extended.csv", "huggingface_nlp_open_weight_models.csv",
    "diffusers_open_weight_models_extended.csv", "diffusers_open_weight_models.csv",
]
REDUNDANT = "hf_open_weight_models.csv"

FORMAT_FILE = {
    "safetensors": "safetensors", "safetensors+gguf": "safetensors",  # has both; safetensors is the graph source
    "gguf": "gguf", "pytorch_bin": "pytorch", "pytorch_pickle": "pytorch", "onnx": "onnx",
    "tf_h5": "other", "flax_msgpack": "other",
}

MULTIMODAL_HINTS = ("Multimodal", "VLM", "Unified multimodal", "+ vision")


def category(family: str, multimodal: bool) -> str:
    f = family
    if f.startswith("Unknown"):
        return "unknown"
    if f in ("spaCy pipeline", "fastText (linear n-gram)") or f.startswith(("Flair", "Stanza", "CTranslate2")):
        return "classic_nlp"  # not tensor-program models; kept so nothing is dropped
    if f.startswith("Draft model"):
        return "speculative_draft"
    if f.startswith("Diffusion LM"):
        return "diffusion_lm"
    if f == "SSM/RNN":
        return "ssm_rnn"
    if f.startswith("Encoder-Decoder"):
        return "encoder_decoder"
    if f.startswith("Encoder"):
        return "encoder"
    if f.startswith("Vision/"):
        if "Multimodal" in f:
            return "vision_multimodal"
        if f in ("Vision/CNN", "Vision/Efficient/mobile"):
            return "vision_cnn"
        if f == "Vision/Hybrid conv/attention":
            return "vision_hybrid"
        return "vision_vit"  # ViT + MLP/token mixer: patch tokens through blocks
    if f.startswith("UNet"):
        return "diffusion_unet"
    if "DiT" in f:
        return "diffusion_dit"
    if f == "VAE":
        return "diffusion_vae"
    if f.startswith(("ControlNet", "Adapter")):
        return "diffusion_control_adapter"
    if f.startswith(("Unified LLM/VLM + diffusion", "Diffusion policy")):
        return "diffusion_multimodal"
    if any(h in f for h in MULTIMODAL_HINTS):
        return "multimodal_vlm"
    if f.startswith("Dense"):
        return "multimodal_vlm" if multimodal else "decoder_dense"
    if f.startswith("Hybrid"):
        base = "decoder_hybrid_moe" if "MoE" in f else "decoder_hybrid"
        return "multimodal_vlm" if multimodal else base
    if f.startswith("MoE"):
        return "multimodal_vlm" if multimodal else "decoder_moe"
    raise ValueError(f"unmapped architecture_family: {family!r}")


# ---------------------------------------------------------------- notes -> structured columns
_NUM = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def _count(s: str) -> str:
    m = re.fullmatch(r"([\d.]+)\s*([KMBT]?)", s.strip())
    return str(int(round(float(m[1]) * _NUM.get(m[2], 1)))) if m else ""


def parse_notes(notes: str) -> dict[str, str]:
    """Pull graph-relevant facts out of the free-text notes column."""
    out: dict[str, str] = {}
    n = notes

    def grab(key, pattern, conv=lambda x: x):
        m = re.search(pattern, n)
        if m and not out.get(key):
            out[key] = conv(m[1])

    grab("model_type", r"(?:^|; )model_type=([\w\-.]+)")
    grab("lm_model_type", r"\(LM=([\w\-.]+)\)")
    grab("architectures", r"(?:^|; )arch=([\w\-.]+)")
    grab("gguf_architecture", r"gguf arch=([\w\-.]+)")
    grab("tasks", r"(?:^|; )tasks?=([^;]+)", lambda x: x.replace(" (untagged)", "").strip())
    grab("params_total", r"(?:^|; )params=([\d.]+\s*[KMBT]?)", _count)
    grab("dtype", r"(?:^|; )dtype=([\w\-]+)")
    grab("num_layers", r"\bL=(\d+)")
    grab("hidden_size", r"\bd=(\d+)")
    grab("num_attention_heads", r"\bH=(\d+)")
    grab("num_kv_heads", r"\bKV=(\d+)")
    grab("routed_experts", r"experts=(\d+)")
    grab("experts_per_token", r"experts=\d+ top(\d+)")
    grab("shared_experts", r"experts=\d+[^;]*shared=(\d+)")
    grab("quant_method", r"quantized=([\w\-]+)")
    grab("variants", r"variants=([^;]+)")
    grab("license", r"license=([^;]+)")
    grab("gated", r"gated=([^;]+)")
    grab("weight_files", r"\((\d+) shards\)")
    grab("weight_files", r"(\d+) GGUF file\(s\)")
    grab("timm_module", r"timm module=([\w\-.]+)")
    grab("pretrained_tag", r"pretrained_tag=([\w\-.]+)")
    grab("pipeline_class", r"pipeline=([\w\-.]+)")
    grab("backbone_class", r"backbone=([\w\-.]+)")
    grab("components", r"components=([^;]+)")
    grab("metadata_files", r"inspect ([^;]+)", lambda x: re.sub(r"\s*\(\d+ shards?\)", "", x)
         .replace(" + ", "|").replace(", ", "|").replace(" and ", "|").strip())
    if "timm module=" in n and not out.get("metadata_files"):
        out["metadata_files"] = "config.json|model.safetensors"

    flags = []
    for flag, pat in [("multimodal", r"multimodal:"), ("sliding_global_attention", r"sliding/global"),
                      ("mla", r"\bMLA\b"), ("linear_attn_ssm", r"linear-attn/SSM"),
                      ("trust_remote_code", r"custom modeling code"), ("pickle_weights", r"pickle weights"),
                      ("single_file", r"single-file"), ("gguf_llamacpp_names", r"llama\.cpp scheme")]:
        if re.search(pat, n):
            flags.append(flag)
    out["graph_flags"] = "|".join(flags)
    return out


def metadata_origin(row: dict) -> str:
    """Where a graph builder must read hyperparameters from."""
    src = row.get("family_source", "")
    if src:
        return src
    fmt = row.get("target_format", "")
    if fmt == "gguf":
        return "GGUF header"
    if row.get("backbone_config"):
        return row["backbone_config"]
    return "config.json" if row.get("metadata_files") or row.get("model_type") else ""


# ---------------------------------------------------------------- merge
LEAD = ["graph_category", "huggingface_model_id", "huggingface_url", "model_series", "target_format",
        "architecture_family", "model_type", "lm_model_type", "architectures", "gguf_architecture", "tasks",
        "params_total", "dtype", "num_layers", "hidden_size", "num_attention_heads", "num_kv_heads",
        "routed_experts", "shared_experts", "experts_per_token", "graph_flags", "metadata_origin", "metadata_files"]
TAIL = ["families_seen", "sources", "notes"]
_DIGITS = ("params_total", "num_layers", "hidden_size", "num_attention_heads", "num_kv_heads",
           "routed_experts", "shared_experts", "experts_per_token")


def load() -> tuple[dict[str, dict], list[str]]:
    merged: dict[str, dict] = {}
    columns: list[str] = []
    for name in SOURCES:
        with open(SRC / name, newline="") as fh:
            for raw in csv.DictReader(fh):
                rid = raw["huggingface_model_id"].strip()
                row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
                # nlp_extended "params" is a raw count, deepseek has params_total; "params_h" is the human form.
                if row.get("params", "").isdigit():
                    row["params_total"] = row.pop("params")
                row.update({k: v for k, v in parse_notes(row.get("notes", "")).items() if not row.get(k)})
                for k in row:
                    if k not in columns:
                        columns.append(k)
                m = merged.setdefault(rid, {"sources": [], "families_seen": [], "notes_all": []})
                m["sources"].append(name.removesuffix(".csv"))
                fam = row.get("architecture_family", "")
                if fam and fam not in m["families_seen"]:
                    m["families_seen"].append(fam)
                note = row.pop("notes", "")
                if note and note not in m["notes_all"]:
                    m["notes_all"].append(note)
                if row.get("graph_flags") and m.get("graph_flags"):  # flags: union across sources
                    m["graph_flags"] = "|".join(dict.fromkeys(m["graph_flags"].split("|") + row.pop("graph_flags").split("|")))
                for k, v in row.items():
                    if v and not m.get(k):
                        m[k] = v
    for m in merged.values():
        mm = "multimodal" in m.get("graph_flags", "").split("|") or any(
            any(h in f for h in MULTIMODAL_HINTS) for f in m["families_seen"])
        m["graph_category"] = category(m["architecture_family"], mm)
        m["metadata_origin"] = metadata_origin(m)
        m["notes"] = " || ".join(m.pop("notes_all"))
        m["sources"] = "|".join(m["sources"])
        m["families_seen"] = "|".join(m["families_seen"])
        for k in _DIGITS:
            if m.get(k) and not m[k].isdigit():
                m[k] = _count(m[k]) or m[k]
    drop = {"params", "notes"}
    order = LEAD + [c for c in columns if c not in LEAD and c not in TAIL and c not in drop] + TAIL
    return merged, order


# ---------------------------------------------------------------- metadata specs (what feeds the graph JSON dims)
# "dims": output dim symbol (json/*.json `dims`) -> [safetensors config.json expression, GGUF KV expression].
# "attrs": other hyperparameters the builder needs (layer count, eps, rope, routing...), same [config, gguf] form.
# <arch> = value of general.architecture. B, T, q, k are example sizes chosen by the builder, not model metadata.
SAFETENSORS_COMMON = {
    "files": ["config.json", "model.safetensors.index.json (sharded: weight_map tensor->file)",
              "*.safetensors header: 8-byte LE length + JSON {name: {dtype, shape, data_offsets}, __metadata__}"],
    "tensor_fields": ["name -> tensors[].checkpoint_key", "shape -> tensors[].shape", "dtype -> tensors[].dtype"],
    "quantization": "config.json quantization_config {quant_method, weight_block_size, fmt}; scale tensors *_scale_inv / *.scales",
}
GGUF_COMMON = {
    "files": ["*.gguf header: magic, version, tensor_count, kv_count, KV metadata, tensor infos"],
    "kv": ["general.architecture", "general.name", "general.file_type", "general.quantization_version",
           "general.alignment", "tokenizer.ggml.model", "tokenizer.ggml.tokens"],
    "tensor_fields": ["name (llama.cpp scheme: token_embd, blk.N.attn_q, blk.N.ffn_gate_exps, output, output_norm) "
                      "-> tensors[].checkpoint_key, kept as-is (see json/mixtral-8x7b-instruct-q4km.json)",
                      "dims (reversed vs PyTorch order) -> shape", "ggml type (Q4_K, Q8_0, F16, ...) -> dtype"],
    "split_files": "multi-part *-00001-of-0000N.gguf: tensors spread across parts; KV in part 1",
}
_HQ = "num_attention_heads | n_head"
_HKV = "num_key_value_heads (default = Hq)"
_DH = "head_dim (default hidden_size/num_attention_heads)"
DECODER_DIMS = {
    "D": ["hidden_size | n_embd (GPT-2)", "<arch>.embedding_length"],
    "V": ["vocab_size", "<arch>.vocab_size | len(tokenizer.ggml.tokens)"],
    "F": ["intermediate_size | n_inner (GPT-2; null = 4*D)", "<arch>.feed_forward_length"],
    "H": [f"GQA (Hkv present): ({_HQ}) / ({_HKV}) = query heads per KV group; MHA/GPT-2: {_HQ}",
          "<arch>.attention.head_count / <arch>.attention.head_count_kv (GQA); <arch>.attention.head_count (MHA)"],
    "Hkv": [_HKV, "<arch>.attention.head_count_kv"],
    "Dh": [_DH, "<arch>.attention.key_length (else embedding_length / attention.head_count)"],
    "Dh2": ["= Dh (second label of the same size, for q.k^T score axes)", "= Dh"],
    "Dq": ["Hq * Dh", "<arch>.attention.head_count * Dh"],
    "Dkv": ["Hkv * Dh", "<arch>.attention.head_count_kv * Dh"],
    "P": ["max_position_embeddings | n_positions (GPT-2; learned position table)", "<arch>.context_length"],
    "Dc": ["= D (GPT-2 Conv1D projection width)", "= D"],
    "QKV": ["3 * D (GPT-2 fused c_attn)", "attn_qkv tensor shape"],
}
DECODER_ATTRS = {
    "L": ["num_hidden_layers | n_layer (GPT-2)", "<arch>.block_count"],
    "eps": ["rms_norm_eps | layer_norm_eps | layer_norm_epsilon",
            "<arch>.attention.layer_norm_rms_epsilon | <arch>.attention.layer_norm_epsilon"],
    "rope": ["rope_theta, rope_scaling, partial_rotary_factor",
             "<arch>.rope.freq_base, <arch>.rope.scaling.type, <arch>.rope.dimension_count"],
    "window": ["sliding_window, layer_types (sliding/full)",
               "<arch>.attention.sliding_window, <arch>.attention.sliding_window_pattern"],
    "tied": ["tie_word_embeddings -> aliases {lm_head.weight: E}", "no output.weight tensor -> alias to token_embd.weight"],
}
MOE_DIMS = {
    "E": ["num_local_experts | n_routed_experts | num_experts", "<arch>.expert_count"],
    "Fe": ["moe_intermediate_size (Mixtral: intermediate_size)",
           "<arch>.expert_feed_forward_length (else <arch>.feed_forward_length)"],
}
MOE_ATTRS = {
    "topk": ["num_experts_per_tok", "<arch>.expert_used_count"],
    "shared": ["n_shared_experts, shared_expert_intermediate_size",
               "<arch>.expert_shared_count, <arch>.expert_shared_feed_forward_length"],
    "dense_layers": ["first_k_dense_replace | mlp_only_layers | decoder_sparse_step", "<arch>.leading_dense_block_count"],
    "router": ["scoring_func, topk_method, n_group, topk_group, routed_scaling_factor, norm_topk_prob",
               "<arch>.expert_gating_func, <arch>.expert_group_count, <arch>.expert_group_used_count, "
               "<arch>.expert_weights_scale, <arch>.expert_weights_norm"],
}
MLA_DIMS = {  # json/deepseek-v3.json; H here is the total head count
    "H": ["num_attention_heads", "<arch>.attention.head_count"],
    "QR": ["q_lora_rank", "<arch>.attention.q_lora_rank"],
    "Dn": ["qk_nope_head_dim", "Dqk - Dr"],
    "Dr": ["qk_rope_head_dim", "<arch>.rope.dimension_count"],
    "Dr2": ["= Dr", "= Dr"],
    "Dqk": ["Dn + Dr", "<arch>.attention.key_length_mla (older files: <arch>.attention.key_length)"],
    "QH": ["H * Dqk", "H * Dqk"],
    "KR": ["kv_lora_rank", "<arch>.attention.kv_lora_rank"],
    "KA": ["KR + Dr (kv_a_proj_with_mqa width)", "KR + Dr"],
    "Dv": ["v_head_dim", "<arch>.attention.value_length_mla (older files: <arch>.attention.value_length)"],
    "Dkvh": ["Dn + Dv", "Dn + Dv"],
    "KH": ["H * Dkvh", "H * Dkvh"],
    "HV": ["H * Dv", "H * Dv"],
}
MLA_ATTRS = {"mtp": ["num_nextn_predict_layers", "<arch>.nextn_predict_layers"],
             "dsa_indexer": ["index_topk, index_n_heads, index_head_dim",
                             "<arch>.attention.indexer.head_count, <arch>.attention.indexer.key_length"]}
HYBRID_DIMS = {  # src/gpt2_pytorch/tensorprog/qwen3_5.py, json/qwen3.8-27b-fp8.json
    "Hk": ["linear_num_key_heads", "<arch>.ssm.group_count"],
    "Hv": ["linear_num_value_heads", "<arch>.ssm.time_step_rank"],
    "Dk": ["linear_key_head_dim", "<arch>.ssm.state_size"],
    "Dv": ["linear_value_head_dim", "<arch>.ssm.inner_size / Hv"],
    "Kc": ["linear_conv_kernel_dim | conv_kernel | d_conv", "<arch>.ssm.conv_kernel"],
    "Dqk": ["Hk * Dk", "Hk * Dk"],
    "Dvv": ["Hv * Dv", "<arch>.ssm.inner_size"],
    "C": ["2*Hk*Dk + Hv*Dv (conv channels)", "2*Hk*Dk + Hv*Dv"],
    "R": ["Hv / Hk", "Hv / Hk"],
    "U": ["1 (unit axis)", "1"],
    "G": ["num_attention_heads / num_key_value_heads", "<arch>.attention.head_count / <arch>.attention.head_count_kv"],
    "Dqg": ["2 * num_attention_heads * head_dim (q + gate)", "2 * <arch>.attention.head_count * Dh"],
    "QG": ["2 * head_dim", "2 * Dh"],
}
HYBRID_ATTRS = {
    "layer_types": ["layer_types | full_attention_interval | attn_layer_indices", "<arch>.full_attention_interval"],
    "mamba": ["state_size | d_state, expand, n_groups, chunk_size",
              "<arch>.ssm.state_size, <arch>.ssm.inner_size, <arch>.ssm.group_count"],
}
VLM_ATTRS = {
    "text": ["text_config.* (decoder keys; checkpoint prefix model.language_model.)", "main .gguf holds the LM"],
    "vision": ["vision_config: hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, patch_size, "
               "image_size, spatial_merge_size, temporal_patch_size",
               "mmproj-*.gguf: clip.vision.embedding_length, clip.vision.block_count, clip.vision.attention.head_count, "
               "clip.vision.feed_forward_length, clip.vision.patch_size, clip.vision.image_size, "
               "clip.vision.spatial_merge_size, clip.projector_type"],
    "projector": ["projector_hidden_act, image_token_id | image_token_index, mm_projector_type",
                  "clip.vision.projector_type, clip.vision.projection_dim, mm.* tensors"],
    "audio": ["audio_config (if present)", "clip.audio.embedding_length, clip.audio.block_count, clip.audio.projector_type"],
}
ENCODER_DIMS = {
    "D": ["hidden_size", "<arch>.embedding_length"], "H": ["num_attention_heads", "<arch>.attention.head_count"],
    "Dh": ["hidden_size / num_attention_heads", "<arch>.embedding_length / <arch>.attention.head_count"],
    "F": ["intermediate_size", "<arch>.feed_forward_length"], "V": ["vocab_size", "len(tokenizer.ggml.tokens)"],
    "P": ["max_position_embeddings", "<arch>.context_length"],
    "Tt": ["type_vocab_size", "tokenizer.ggml.token_type_count"],
}
ENCODER_ATTRS = {
    "L": ["num_hidden_layers", "<arch>.block_count"],
    "eps": ["layer_norm_eps", "<arch>.attention.layer_norm_epsilon"],
    "head": ["architectures[0] suffix (MaskedLM / TokenClassification / SequenceClassification), num_labels, id2label",
             "<arch>.pooling_type"],
    "pos": ["position_embedding_type; rotary (ModernBERT: global_rope_theta, local_attention)", "<arch>.rope.freq_base"],
}
ENCDEC_DIMS = {
    "D": ["d_model", "<arch>.embedding_length"], "F": ["d_ff | encoder_ffn_dim", "<arch>.feed_forward_length"],
    "H": ["num_heads | encoder_attention_heads", "<arch>.attention.head_count"],
    "Dh": ["d_kv (T5) else d_model/H", "<arch>.attention.key_length"],
    "V": ["vocab_size", "len(tokenizer.ggml.tokens)"],
}
ENCDEC_ATTRS = {
    "L_enc": ["num_layers | encoder_layers", "<arch>.block_count"],
    "L_dec": ["num_decoder_layers | decoder_layers", "<arch>.decoder_block_count (else block_count)"],
    "relpos": ["relative_attention_num_buckets, relative_attention_max_distance", "<arch>.attention.relative_buckets_count"],
    "act": ["feed_forward_proj (gated-gelu / relu)", "infer from ffn_gate tensor presence"],
    "start": ["decoder_start_token_id", "<arch>.decoder_start_token_id"],
}
DIFFUSION_COMMON = {
    "files": ["model_index.json (pipeline: component -> [library, class])", "<component>/config.json",
              "<component>/diffusion_pytorch_model.safetensors(.index.json)"],
    "gguf_note": "diffusion GGUF (city96 style) carries general.architecture (flux, sd1, sdxl, sd3, wan, ...) and "
                 "tensor infos only: derive dims from tensor shapes",
    "single_file": "single-file checkpoints (original layout keys: model.diffusion_model.*): derive dims from tensor keys/shapes",
}
TIMM = "timm.create_model(arch, pretrained=False) + torch.fx / torch.export; dims from safetensors header shapes"

SPECS = {
    "decoder_dense": {"graph": "decoder-only transformer: embed -> L x (norm, attention, norm, MLP) -> norm -> lm_head",
                      "dims": DECODER_DIMS, "attrs": DECODER_ATTRS,
                      "examples": ["json/distilgpt2.json", "json/mistral-7b-instruct-v0.3.json", "json/qwen2.5-7b-instruct.json",
                                   "json/gemma-2-2b-it.json", "json/phi-3.5-mini-instruct.json"]},
    "decoder_moe": {"graph": "decoder-only transformer with routed expert MLP (router -> topk -> experts [+ shared]); "
                             "MLA attention when graph_flags has mla / kv_lora_rank is set (DeepSeek)",
                    "dims": {**DECODER_DIMS, **MOE_DIMS}, "dims_mla": MLA_DIMS,
                    "attrs": {**DECODER_ATTRS, **MOE_ATTRS, **MLA_ATTRS},
                    "examples": ["json/mixtral-8x7b-instruct-q4km.json"], "examples_mla": ["json/deepseek-v3.json"]},
    "decoder_hybrid": {"graph": "decoder mixing full attention with linear-attention / SSM layers (layer_types decides per layer)",
                       "dims": {**DECODER_DIMS, **HYBRID_DIMS}, "attrs": {**DECODER_ATTRS, **HYBRID_ATTRS},
                       "examples": ["json/qwen3.8-27b-fp8.json"]},
    "decoder_hybrid_moe": {"graph": "hybrid attention/SSM decoder with routed expert MLP",
                           "dims": {**DECODER_DIMS, **HYBRID_DIMS, **MOE_DIMS},
                           "attrs": {**DECODER_ATTRS, **HYBRID_ATTRS, **MOE_ATTRS}},
    "multimodal_vlm": {"graph": "vision (and/or audio) encoder -> projector -> decoder LM; pick the LM dims block from "
                                "model_type / lm_model_type / graph_flags (dense, moe, mla, hybrid)",
                       "dims": DECODER_DIMS, "dims_moe": MOE_DIMS, "dims_mla": MLA_DIMS, "dims_hybrid": HYBRID_DIMS,
                       "attrs": {**DECODER_ATTRS, **MOE_ATTRS, **HYBRID_ATTRS, **VLM_ATTRS},
                       "examples": ["json/qwen3.8-27b-fp8.json (text tower only; checkpoint prefix model.language_model.)"]},
    "speculative_draft": {"graph": "small draft head (few decoder layers) that consumes the target model's hidden states",
                          "dims": DECODER_DIMS,
                          "attrs": {**DECODER_ATTRS, "target": ["target model from repo name / base_model", "-"]}},
    "diffusion_lm": {"graph": "bidirectional (non-causal) transformer denoiser over tokens; dense or MoE",
                     "dims": {**DECODER_DIMS, **MOE_DIMS},
                     "attrs": {**DECODER_ATTRS, **MOE_ATTRS, "mask": ["mask_token_id", "tokenizer.ggml.mask_token_id"]}},
    "ssm_rnn": {"graph": "recurrent / state-space stack (Mamba, RWKV, xLSTM) without attention",
                "dims": {k: DECODER_DIMS[k] for k in ("D", "V")},
                "attrs": {"L": DECODER_ATTRS["L"], **HYBRID_ATTRS,
                          "rwkv": ["attention_hidden_size, head_size", "<arch>.wkv.head_size, <arch>.time_mix_extra_dim"]}},
    "encoder": {"graph": "bidirectional encoder (BERT / RoBERTa / DeBERTa / ModernBERT ...) + task head",
                "dims": ENCODER_DIMS, "attrs": ENCODER_ATTRS},
    "encoder_decoder": {"graph": "encoder stack + decoder stack with cross-attention (T5, BART, Marian, ...)",
                        "dims": {**ENCDEC_DIMS, **MOE_DIMS}, "attrs": {**ENCDEC_ATTRS, **MOE_ATTRS}},
    "diffusion_unet": {"graph": "UNet: conv_in -> down blocks (resnet + [cross-]attention) -> mid -> up blocks (skips) -> conv_out",
                       "attrs": {"C_in": ["in_channels", "tensor shape"], "C_out": ["out_channels", "tensor shape"],
                                 "blocks": ["block_out_channels, down_block_types, up_block_types, layers_per_block", "tensor keys"],
                                 "attn": ["attention_head_dim | num_attention_heads, cross_attention_dim, "
                                          "transformer_layers_per_block", "tensor shapes"],
                                 "S": ["sample_size", "-"],
                                 "cond": ["addition_embed_type, projection_class_embeddings_input_dim", "-"]},
                       **DIFFUSION_COMMON},
    "diffusion_dit": {"graph": "patchify -> N x transformer blocks (adaLN; MMDiT: joint text/image streams) -> unpatchify",
                      "attrs": {"L": ["num_layers (+ num_single_layers for Flux)", "count block tensors"],
                                "H": ["num_attention_heads", "tensor shapes"], "Dh": ["attention_head_dim", "tensor shapes"],
                                "D": ["H*Dh | hidden_size | inner_dim", "tensor shapes"], "C_in": ["in_channels", "-"],
                                "patch": ["patch_size (+ patch_size_t for video)", "-"],
                                "text": ["joint_attention_dim | caption_channels | text_dim, pooled_projection_dim", "-"]},
                      **DIFFUSION_COMMON},
    "diffusion_vae": {"graph": "conv encoder -> latent (mean, logvar) -> conv decoder",
                      "attrs": {"C_lat": ["latent_channels", "tensor shape"], "blocks": ["block_out_channels, layers_per_block", "-"],
                                "scale": ["scaling_factor, shift_factor", "-"]}, **DIFFUSION_COMMON},
    "diffusion_control_adapter": {"graph": "ControlNet (copy of UNet/DiT encoder + zero convs) or adapter (T2I / IP / Motion) "
                                           "feeding a base model",
                                  "attrs": {"base": ["base_model, conditioning_channels, conditioning_embedding_out_channels", "-"]},
                                  **DIFFUSION_COMMON},
    "diffusion_multimodal": {"graph": "LLM/VLM backbone + diffusion head (unified generation, VLA action diffusion)",
                             "attrs": {**VLM_ATTRS, "head": ["diffusion head config in components", "-"]}, **DIFFUSION_COMMON},
    "vision_cnn": {"graph": "timm CNN: stem -> stages of conv blocks -> pool -> classifier", "builder": TIMM,
                   "attrs": {"arch": ["config.json architecture + model_args", "-"], "classes": ["num_classes", "-"],
                             "features": ["num_features", "-"], "input": ["pretrained_cfg.input_size, mean, std", "-"]}},
    "vision_vit": {"graph": "patch embed -> [cls/pos] -> L x transformer (or MLP-mixer) blocks -> pool -> head", "builder": TIMM,
                   "attrs": {"patch": ["model_args patch_size", "patch_embed.proj.weight shape"],
                             "D": ["embed_dim", "cls_token / pos_embed shape"], "L": ["depth", "count blocks.N"],
                             "H": ["num_heads", "-"], "input": ["pretrained_cfg.input_size", "-"],
                             "classes": ["num_classes", "head.weight shape"]}},
    "vision_hybrid": {"graph": "conv stages + attention stages (CoAtNet, MaxViT, EfficientFormer, ...)", "builder": TIMM,
                      "attrs": {"arch": ["config.json architecture + model_args", "-"], "input": ["pretrained_cfg.input_size", "-"]}},
    "vision_multimodal": {"graph": "image tower (+ text tower for CLIP/SigLIP) -> projection to shared embedding",
                          "builder": "open_clip / timm config + torch.export",
                          "attrs": {"vision": ["vision_cfg | vision_config", "-"], "text": ["text_cfg | text_config", "-"],
                                    "embed": ["embed_dim | projection_dim", "-"]}},
    "classic_nlp": {"graph": "not a tensor-program model (spaCy, Flair, fastText, Stanza, CTranslate2 export); "
                             "CTranslate2 rows point at a source model that can be graphed instead"},
    "unknown": {"graph": "family not readable from configs: derive from safetensors/GGUF tensor names and shapes, "
                         "then move the row to a real category"},
}


def spec_for(cat: str, formats: Counter) -> dict:
    s = {"category": cat, "row_counts_by_format": dict(sorted(formats.items())), **SPECS[cat],
         "dims_format": "symbol -> [safetensors config.json expression, GGUF KV expression]; <arch> = general.architecture; "
                        "B, T, q, k are example sizes picked by the builder"}
    if not cat.startswith(("diffusion_", "vision_", "classic", "unknown")):
        s["safetensors"] = SAFETENSORS_COMMON
        s["gguf"] = GGUF_COMMON
    elif cat.startswith("vision_"):
        s["safetensors"] = SAFETENSORS_COMMON
    s["graph_json"] = ("tensormorph-viz-graph v1 (json/*.json): model.name, dims, axes, aliases, tensors[] "
                       "(checkpoint_key, shape, dtype), ops[], edges[], groups, multiplications, totals")
    return s


def check_specs() -> None:
    """Spec dims must cover every dims symbol of the example graphs; GGUF keys must exist in llama.cpp's gguf package."""
    for cat, s in SPECS.items():
        have = set().union(*(v for k, v in s.items() if k.startswith("dims") and isinstance(v, dict)))
        for ex in s.get("examples", []) + s.get("examples_mla", []):
            dims = set(json.loads((ROOT / ex.split(" ")[0]).read_text())["dims"]) - {"B", "T", "q", "k"}
            assert dims <= have, f"{cat}: {ex} dims missing from spec: {sorted(dims - have)}"
    try:
        from gguf.constants import Keys
    except ImportError:
        print("skipped GGUF key validation (run: uv run --with gguf python tools/split_model_catalog.py --check)")
        return
    known: set[str] = set()
    stack = [Keys]
    while stack:
        for v in vars(stack.pop()).values():
            if isinstance(v, str):
                known.add(v.replace("{arch}", "<arch>"))
            elif isinstance(v, type):
                stack.append(v)
    text = json.dumps([SPECS, GGUF_COMMON])
    used = set(re.findall(r"(?:<arch>|tokenizer\.ggml|clip|general)\.[a-z_.]*[a-z_]", text))
    missing = sorted(k for k in used if k not in known)
    assert not missing, f"GGUF keys not in gguf.constants.Keys: {missing}"
    print(f"GGUF keys ok: {len(used)} checked against gguf.constants.Keys")


# ---------------------------------------------------------------- write
def write(merged: dict[str, dict], order: list[str]) -> dict[str, Counter]:
    for d in OUT.glob("*/"):  # only generated category dirs; README.md and other top-level files stay
        if (d / "metadata_spec.json").exists():
            shutil.rmtree(d)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in merged.values():
        buckets[(m["graph_category"], FORMAT_FILE[m["target_format"]])].append(m)
    counts: dict[str, Counter] = defaultdict(Counter)
    for (cat, fmt), rows in sorted(buckets.items()):
        rows.sort(key=lambda r: (r.get("model_series", "").lower(), r["huggingface_model_id"].lower()))
        cols = [c for c in order if any(r.get(c) for r in rows)]  # drop columns empty for the whole file
        d = OUT / cat
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{fmt}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        counts[cat][fmt] = len(rows)
    for cat, fc in counts.items():
        (OUT / cat / "metadata_spec.json").write_text(json.dumps(spec_for(cat, fc), indent=2) + "\n")
    with open(OUT / "all_models.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=order, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(merged.values(), key=lambda r: (r["graph_category"], r["huggingface_model_id"].lower())))
    with open(OUT / "index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "format", "rows", "file"])
        for cat in sorted(counts):
            for fmt, n in sorted(counts[cat].items()):
                w.writerow([cat, fmt, n, f"{cat}/{fmt}.csv"])
    return counts


def check(merged: dict[str, dict]) -> None:
    ids = set()
    for name in SOURCES + [REDUNDANT]:
        with open(SRC / name, newline="") as fh:
            ids |= {r["huggingface_model_id"].strip() for r in csv.DictReader(fh)}
    assert ids == set(merged), f"id mismatch: {len(ids ^ set(merged))}"
    seen: Counter = Counter()
    for f in OUT.glob("*/*.csv"):
        with open(f, newline="") as fh:
            seen.update(r["huggingface_model_id"] for r in csv.DictReader(fh))
    assert set(seen) == ids and max(seen.values()) == 1, "every id must land in exactly one category/format file"
    assert sum(seen.values()) == len(ids)
    assert "databricks/dbrx-instruct" in seen
    dup = [k for k, n in Counter(i.lower() for i in ids).items() if n > 1]  # HF ids are case-insensitive
    assert not dup, f"case-only duplicate ids: {dup[:10]}"
    ds, js = merged["deepseek-ai/DeepSeek-V3"], json.loads((ROOT / "json/deepseek-v3.json").read_text())["dims"]
    got = {"D": ds["hidden_size"], "H": ds["num_attention_heads"], "QR": ds["q_lora_rank"], "KR": ds["kv_lora_rank"],
           "E": ds["routed_experts"], "V": ds["vocab_size"]}
    assert {k: int(v) for k, v in got.items()} == {k: js[k] for k in got}, (got, js)
    with open(SRC / REDUNDANT, newline="") as fh:  # skipping the redundant file must lose nothing
        assert {r["huggingface_model_id"] for r in csv.DictReader(fh)} <= set(merged)
    m = merged["deepseek-ai/dflash_qwen3_4b_block7"]
    assert m["graph_category"] == "speculative_draft" and m["num_layers"] == "5" and m["notes"], m
    r = merged["Yale-CompBio/ablang2-hf"]
    assert (r["num_layers"], r["hidden_size"], r["num_attention_heads"]) == ("12", "480", "20"), r
    assert parse_notes("params=7.25B; experts=384 top6 shared=1")["experts_per_token"] == "6"
    assert _count("7.25B") == "7250000000" and _count("45M") == "45000000"
    print(f"check ok: {len(ids)} ids, each in exactly one file")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="run the self-check after writing")
    args = ap.parse_args()
    merged, order = load()
    counts = write(merged, order)
    total = 0
    for cat in sorted(counts):
        n = sum(counts[cat].values())
        total += n
        print(f"{cat:28} {n:6}  " + "  ".join(f"{k}={v}" for k, v in sorted(counts[cat].items())))
    print(f"{'TOTAL':28} {total:6}  -> {OUT.relative_to(ROOT)}")
    if args.check:
        check(merged)
        check_specs()


if __name__ == "__main__":
    main()
