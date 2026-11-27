"""Compatibility helpers that tolerate varied positional calling styles."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def looks_like_package_dict(value: Any) -> bool:
    return isinstance(value, dict) and (
        "groups" in value or "weights" in value and "weight_version" in value
    )


def classify_operator_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map heterogeneous positional args to the operator's keyword names."""

    mapped = dict(kwargs)
    positionals: list[Any] = []

    for arg in args:
        if arg is None:
            continue
        # Explicit keyword-ish objects detected by class name to avoid cycles.
        cls_name = type(arg).__name__
        module = type(arg).__module__

        if cls_name == "WeightPackage":
            mapped.setdefault("package", arg)
        elif isinstance(arg, (str, Path)) and Path(str(arg)).exists():
            mapped.setdefault("package", arg)
        elif looks_like_package_dict(arg):
            mapped.setdefault("package", arg)
        elif cls_name in ("GroupLayout",) or (
            isinstance(arg, dict) and "group_axis" in arg
        ):
            mapped.setdefault("group_layout", arg)
        elif cls_name in ("ActivationStrategy", "ActivationPolicy"):
            mapped.setdefault("activation_strategy", arg)
        elif cls_name in ("ErrorBudget",):
            mapped.setdefault("error_budget", arg)
        elif isinstance(arg, (list, tuple)) and arg and all(
            isinstance(item, dict) and ("scale" in item or "values" in item)
            for item in arg[:1]
        ):
            mapped.setdefault("groups", list(arg))
        elif isinstance(arg, dict):
            # Ambiguous dict: strategy-ish vs budget-ish vs layout.
            if any(k in arg for k in ("low", "high", "activation_range", "name")):
                mapped.setdefault("activation_strategy", arg)
            elif any(k in arg for k in ("max_abs_error", "atol", "tolerance")):
                mapped.setdefault("error_budget", arg)
            else:
                positionals.append(arg)
        elif isinstance(arg, bool):
            positionals.append(arg)
        elif isinstance(arg, int):
            mapped.setdefault("group_size", arg)
        elif isinstance(arg, float):
            mapped.setdefault("error_budget", arg)
        elif isinstance(arg, str):
            # Non-path string: version or profile/activation name.
            if arg in ("preview", "ranking", "fast", "fine_rank", "rerank"):
                mapped.setdefault("profile", arg)
            elif arg in ("relu", "hardtanh", "identity", "none", "gelu", "tanh", "sigmoid"):
                mapped.setdefault("activation_strategy", arg)
            else:
                mapped.setdefault("weight_version", arg)
        else:
            positionals.append(arg)

    if positionals and "package" not in mapped:
        mapped["package"] = positionals[0]
    return mapped


def classify_service_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    mapped = dict(kwargs)
    for arg in args:
        cls_name = type(arg).__name__
        if isinstance(arg, (list, dict)):
            mapped.setdefault("packages", arg)
        elif cls_name == "ResultCache":
            mapped.setdefault("cache", arg)
        elif cls_name == "ResultRegistry":
            mapped.setdefault("registry", arg)
        elif cls_name == "ErrorBudget":
            mapped.setdefault("error_budget", arg)
        elif isinstance(arg, str):
            mapped.setdefault("error_budget", arg) if False else None
    return mapped
