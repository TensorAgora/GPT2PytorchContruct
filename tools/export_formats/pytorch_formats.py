"""PyTorch family: state_dicts (.pt/.pth/.bin, zip and legacy pickle), training checkpoints (.pt, Lightning-style .ckpt),
full-model pickle, TorchScript, torch.export (.pt2), Safetensors. Each file is loaded back and its logits compared."""
import copy
import shutil
from dataclasses import asdict

import torch
from safetensors.torch import load_file, save_file

from gpt2_pytorch import DistilGPT2LMHeadModel
from gpt2_pytorch.checkpoint import DEFAULT_CHECKPOINT
from gpt2_pytorch.checkpoint.loader import load_state_dict_into

from .common import Manifest, compare, hf_state_dict, load_source, parse_args, reference_logits

args = parse_args(__doc__)
model, ids = load_source(args.model)
ref = reference_logits(model, ids)
hf = hf_state_dict(args.model, model)
out, m, name = args.out / "pytorch", Manifest(args, "pytorch"), args.model
out.mkdir(exist_ok=True)


def logits_of(state, hf_names=True):
    fresh = DistilGPT2LMHeadModel(model.config).eval()
    load_state_dict_into(fresh, state) if hf_names else fresh.load_state_dict(state)
    return compare(ref, reference_logits(fresh, ids))


def state_dict_ours():
    p = out / f"{name}.state_dict.pt"
    torch.save(model.state_dict(), p)
    return [p], logits_of(torch.load(p, weights_only=True), hf_names=False)


def state_dict_hf_pth():
    p = out / f"{name}.hf_state_dict.pth"
    torch.save(hf, p)
    return [p], logits_of(torch.load(p, weights_only=True))


def bin_zip():
    p = out / "pytorch_model.bin"
    torch.save(hf, p)
    return [p], logits_of(torch.load(p, weights_only=True))


def bin_legacy():
    p = out / "pytorch_model.legacy.bin"
    if name == "distilgpt2":
        shutil.copyfile(DEFAULT_CHECKPOINT, p)  # HF's own file: legacy pickle, storages tagged cuda:0
    else:
        torch.save(hf, p, _use_new_zipfile_serialization=False)
    return [p], logits_of(torch.load(p, weights_only=True, map_location="cpu"))


def _training_state():
    """AdamW state after one backward with lr=0: real optimizer tensors, weights unchanged."""
    trained = copy.deepcopy(model).train()
    optimizer = torch.optim.AdamW(trained.parameters(), lr=0.0, weight_decay=0.0)
    trained(ids, labels=ids).loss.backward()
    optimizer.step()
    return trained, optimizer


def training_checkpoint():
    trained, optimizer = _training_state()
    p = out / f"{name}.training_checkpoint.pt"
    # DataParallel/DDP-style "module." prefix, as trained checkpoints often have
    torch.save({"model": {f"module.{k}": v for k, v in trained.state_dict().items()}, "optimizer": optimizer.state_dict(),
                "epoch": 3, "global_step": 1200, "config": asdict(model.config)}, p)
    ckpt = torch.load(p, weights_only=True)
    return [p], logits_of({k.removeprefix("module."): v for k, v in ckpt["model"].items()}, hf_names=False)


def lightning_style_ckpt():
    trained, optimizer = _training_state()
    p = out / f"{name}.lightning_style.ckpt"
    torch.save({"epoch": 3, "global_step": 1200, "state_dict": {f"model.{k}": v for k, v in trained.state_dict().items()},
                "optimizer_states": [optimizer.state_dict()], "lr_schedulers": [], "hyper_parameters": {"model": name}}, p)
    ckpt = torch.load(p, weights_only=True)
    return [p], logits_of({k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}, hf_names=False)


def full_model_pickle():
    p = out / f"{name}.full_model_pickle.pt"
    torch.save(model, p)  # pickles the classes by reference: unsafe to load from an untrusted source
    loaded = torch.load(p, weights_only=False)  # our own file, trusted
    return [p], compare(ref, reference_logits(loaded, ids))


def torchscript():
    p = out / f"{name}.torchscript.pt"
    with torch.no_grad():
        traced = torch.jit.trace(model, (ids,), strict=False)
    traced.save(str(p))
    with torch.no_grad():
        return [p], compare(ref, torch.jit.load(str(p))(ids).numpy())


def torch_export_pt2():
    p = out / f"{name}.pt2"
    with torch.no_grad():
        torch.export.save(torch.export.export(model, (ids,)), str(p))
        return [p], compare(ref, torch.export.load(str(p)).module()(ids).numpy())


def safetensors_hf():
    p = out / "model.safetensors"
    save_file({k: v.contiguous() for k, v in hf.items() if k != "lm_head.weight"}, p,
              metadata={"format": "pt", "tied": "lm_head.weight=transformer.wte.weight"})
    return [p], logits_of(load_file(p))


m.variant("pytorch", "state_dict (our names, zip) .pt", "none", state_dict_ours)
m.variant("pytorch", "state_dict (HF names, zip) .pth", "none", state_dict_hf_pth)
m.variant("pytorch", "training checkpoint .pt", "none", training_checkpoint,
          "{model: state_dict with 'module.' prefix, optimizer: AdamW state, epoch, global_step}")
m.variant("pytorch", "lightning-style .ckpt", "none", lightning_style_ckpt,
          "PyTorch zip pickle with a .ckpt extension; layout mimics PyTorch Lightning (Lightning itself not used)")
m.variant("pytorch", "full model pickle .pt", "pickled classes", full_model_pickle,
          "torch.save(model): needs the Python classes and arbitrary unpickling; an importer must refuse it")
m.variant("torchscript", "torch.jit.trace .pt", "TorchScript code", torchscript)
m.variant("torch.export", "ExportedProgram .pt2", "ATen graph", torch_export_pt2)
m.variant("binary", "pytorch_model.bin (zip pickle)", "none", bin_zip)
m.variant("binary", "pytorch_model.bin (legacy pickle)", "none", bin_legacy,
          "pre-1.6 non-zip pickle" + (" (copy of Hugging Face's original file, storages on cuda:0)" if name == "distilgpt2" else ""))
m.variant("safetensors", "HF names, lm_head omitted (tied)", "none", safetensors_hf)
m.write()
