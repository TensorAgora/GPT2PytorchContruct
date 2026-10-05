import torch
import torch.nn.functional as F
from torch import nn

from ..config import GPT2Config
from ..outputs import CausalLMOutput
from .backbone import GPT2Model


def causal_lm_loss(logits, labels, ignore_index=-100):
    """Next-token cross-entropy. Returns (loss, shift_logits, shift_labels)."""
    vocab_size = logits.size(-1)
    # position t predicts token t+1: drop the last logit and the first label
    shift_logits = logits[:, :-1, :]  # [B, T-1, V]
    shift_labels = labels[:, 1:]  # [B, T-1]
    loss = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),  # [B*(T-1), V]
        shift_labels.reshape(-1),  # [B*(T-1)]
        ignore_index=ignore_index,
    )
    return loss, shift_logits, shift_labels


class DistilGPT2LMHeadModel(nn.Module):
    """GPT2Model + LM head. lm_head.weight IS transformer token embedding weight (true tying)."""

    def __init__(self, config: GPT2Config | None = None):
        super().__init__()
        self.config = config or GPT2Config()
        self.transformer = GPT2Model(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # Same Parameter object, not a copy: one storage, one gradient, listed under two names.
        self.lm_head.weight = self.transformer.token_embedding.weight

    def forward(self, input_ids, labels=None, position_ids=None, return_debug=False):
        """Plain call returns logits [B, T, V]. With labels and/or return_debug returns CausalLMOutput."""
        if return_debug:
            final_hidden, debug = self.transformer(input_ids, position_ids, return_debug=True)
        else:
            final_hidden = self.transformer(input_ids, position_ids)
            debug = None

        # [B, T, C] -> [B, T, V]
        logits = self.lm_head(final_hidden)

        if labels is None and not return_debug:
            return logits

        loss = None
        if labels is not None:
            loss, shift_logits, shift_labels = causal_lm_loss(logits, labels)

        hidden_states = None
        if return_debug:
            debug["model.logits"] = logits
            if labels is not None:
                debug.update({"loss.shift_logits": shift_logits, "loss.shift_labels": shift_labels, "loss.value": loss})
            hidden_states = [debug["embedding.output"]]
            hidden_states += [debug[f"block.{i}.output"] for i in range(self.config.num_layers)]
            hidden_states.append(final_hidden)

        return CausalLMOutput(logits=logits, loss=loss, hidden_states=hidden_states, debug=debug)
