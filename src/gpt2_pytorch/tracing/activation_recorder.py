from dataclasses import dataclass, field

import torch

from ..model.attention import CausalSelfAttention, flatten_attention_debug
from .hooks import flatten_tensors, tensor_names, tensor_stem
from .tensor_stats import fmt_shape, tensor_stats


@dataclass
class ActivationRecord:
    index: int  # call order (pre-hook)
    module: str  # dotted name, "" is the root
    module_type: str
    depth: int
    inputs: list[dict] = field(default_factory=list)  # tensor_stats + "name"
    outputs: list[dict] = field(default_factory=list)
    internals: dict[str, dict] = field(default_factory=dict)  # attention q/k/v/scores/... name -> stats
    exit_index: int = -1  # completion order (forward hook)
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)  # only with keep_tensors=True

    def __str__(self) -> str:
        ins = ", ".join(fmt_shape(s["shape"]) for s in self.inputs)
        outs = ", ".join(fmt_shape(s["shape"]) for s in self.outputs)
        return f"{self.index:03d} {'  ' * self.depth}{self.module_type} {self.module or '<root>'} ({ins}) -> ({outs})"


class ActivationRecorder:
    """Records every module call (and attention internals) with no changes to model code.

        with ActivationRecorder(model) as recorder:
            logits = model(input_ids)
        for record in recorder.records: print(record)

    Stats are always kept; tensors themselves (detached references) only with keep_tensors=True.
    """

    def __init__(self, model: torch.nn.Module, keep_tensors: bool = False):
        self.model = model
        self.keep_tensors = keep_tensors
        self.records: list[ActivationRecord] = []
        self._stack: list[ActivationRecord] = []
        self._handles = []
        self._old_sinks = {}
        self._exits = 0

    def _describe(self, record, direction, tensors):
        names = tensor_names(record.module, direction, len(tensors))
        stats = [tensor_stats(t) | {"name": n} for n, t in zip(names, tensors)]
        if self.keep_tensors:
            record.tensors.update({n: t.detach() for n, t in zip(names, tensors)})
        return stats

    def _pre_hook(self, name):
        def hook(module, args, kwargs):
            record = ActivationRecord(len(self.records), name, type(module).__name__, len(self._stack))
            record.inputs = self._describe(record, "input", flatten_tensors((args, kwargs)))
            self.records.append(record)
            self._stack.append(record)

        return hook

    def _post_hook(self, name):
        def hook(module, args, kwargs, output):
            record = self._stack.pop()
            record.outputs = self._describe(record, "output", flatten_tensors(output))
            record.exit_index = self._exits
            self._exits += 1

        return hook

    def _sink(self, name):
        def sink(debug):
            record = self._stack[-1]  # the attention module is the innermost active call
            for tensor_name, tensor in flatten_attention_debug(tensor_stem(name), debug).items():
                record.internals[tensor_name] = tensor_stats(tensor)
                if self.keep_tensors:
                    record.tensors[tensor_name] = tensor.detach()

        return sink

    def __enter__(self):
        for name, module in self.model.named_modules():
            self._handles.append(module.register_forward_pre_hook(self._pre_hook(name), with_kwargs=True))
            self._handles.append(module.register_forward_hook(self._post_hook(name), with_kwargs=True))
            if isinstance(module, CausalSelfAttention):
                self._old_sinks[name] = module.debug_sink
                module.debug_sink = self._sink(name)
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        for name, module in self.model.named_modules():
            if name in self._old_sinks:
                module.debug_sink = self._old_sinks[name]
        self._handles.clear()
        self._stack.clear()
        return False
