# 07d DESIGN

## Assembly descriptor

```json
{"id": "qwen3_vl", "version": "1",
 "assembly": {"text": "qwen3", "prefix": {"model.": "model.language_model."}, "config": "text_config"},
 "match": {"architectures": ["Qwen3VLForConditionalGeneration"], "...": []},
 "components": [{"name": "visual", "prefix": "model.visual.", "state": "declared_without_graph"}]}
```

`descriptor.load` turns it into the text descriptor with every HF scheme key rewritten by the first matching prefix rule,
the wrapper's components appended, and the decoder's layer selectors reading `config` (the nested text config).
Binding then classes the text tower `bound` and every component-prefixed tensor `component`; any component declared
without a graph makes the status `partial`, and the graph's `components` lists `{name, state, tensors, params}`. MTP
layers (qwen3_5 `mtp.`, DeepSeek-V3 layer L) are components the same way. C3 judges an assembly by its text model's names.

## Keys (multimodal_vlm safetensors rows, catalog)

| key / architectures | rows | descriptor | layout |
| --- | --: | --- | --- |
| qwen3_5_text / Qwen3_5ForConditionalGeneration | 415 | qwen3_5 (06b) | `model.language_model.`, components `model.visual.`, `mtp.` (Qwen3.8: visual 333, mtp 22) |
| qwen3_vl_text + qwen3_vl / Qwen3VLForConditionalGeneration | 105 | **qwen3_vl** | text = qwen3 under `model.language_model.`, component `model.visual.` |
| qwen2_5_vl + qwen2_vl | 93 | **qwen2_5_vl**, **qwen2_5_vl_v2** | text = qwen2 at `model.` + `visual.` (older), or `model.language_model.` + `model.visual.` (newer); binding picks |
| gemma3_text / Gemma3ForConditionalGeneration (+ decoder_dense gemma3 82) | 49 | **gemma3**, **gemma3_v2** | text = gemma3_text under `language_model.model.` (older) or `model.language_model.`; components vision_tower, multi_modal_projector |

Not this wave: qwen3_5_moe_text (128; the qwen3_5 lowering has no MoE), gemma4_text (116), paligemma (76, gated; the
text is gemma2 under `language_model.model.` and fits the same assembly once a fixture is available), qwen4_exp_text
(47), Qwen3_5Model embedding checkpoints (111, no lm_head). speculative_draft repos (12) are standalone draft heads of
many formats: no_graph with an inventory. In-repo draft/MTP heads are components.

Predicted rows unlocked: 105 + 93 + 49 = 247 multimodal_vlm rows, plus gemma3's 82 decoder_dense rows.
