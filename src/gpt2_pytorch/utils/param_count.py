import torch


def _category(name: str) -> str:
    for needle, category in (
        ("lm_head", "lm head"),
        ("token_embedding", "token embeddings"),
        ("position_embedding", "position embeddings"),
        ("attention", "attention"),
        ("mlp", "mlp"),
    ):
        if needle in name:
            return category
    return "layernorm"  # ln_1, ln_2, final_layer_norm


def parameter_breakdown(model: torch.nn.Module) -> dict:
    """Logical references (tied weights counted under every name) vs unique underlying parameters."""
    logical, unique, seen = {}, {}, set()
    for name, param in model.named_parameters(remove_duplicate=False):
        category = _category(name)
        logical[category] = logical.get(category, 0) + param.numel()
        if id(param) not in seen:
            seen.add(id(param))
            unique[category] = unique.get(category, 0) + param.numel()
    return {"logical": logical, "unique": unique, "logical_total": sum(logical.values()), "unique_total": sum(unique.values())}


def format_breakdown(breakdown: dict) -> str:
    rows = ["category              logical        unique"]
    for category, count in breakdown["logical"].items():
        rows.append(f"{category:<18}{count:>12,}{breakdown['unique'].get(category, 0):>14,}")
    rows.append(f"{'total':<18}{breakdown['logical_total']:>12,}{breakdown['unique_total']:>14,}")
    return "\n".join(rows)
