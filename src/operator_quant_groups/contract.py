"""Dict/JSON request handler.

Useful for subprocess or language neutral checks::

    from operator_quant_groups.contract import handle_request
    response = handle_request({"action": "submit", "payload": {...}, ...})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .eligibility import evaluate_eligibility
from .operator import GroupedQuantizedLinear, quantized_linear
from .service import QuantGroupService

# A process wide service lets sequential requests share cache / receipts.
_DEFAULT_SERVICE = QuantGroupService()


def _load_fixture(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "input.json"
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def handle_request(request: dict[str, Any] | str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Handle a plain dict request and return a JSON safe dict.

    Withdrawal / missing-trace conditions are returned as structured
    ``{"ok": False, "error_code": ...}`` responses rather than raised, so the
    JSON/CLI boundary always yields serialisable output.
    """

    try:
        return dispatch_request(request, **kwargs)
    except Exception as exc:  # boundary: serialise failures for callers
        code = getattr(exc, "reason_code", None) or exc.__class__.__name__
        return {
            "ok": False,
            "error": str(exc),
            "error_code": str(code),
            "error_type": exc.__class__.__name__,
        }


def dispatch_request(request: dict[str, Any] | str | None = None, **kwargs: Any) -> dict[str, Any]:
    import json

    if request is None:
        request = {}
    if isinstance(request, str):
        text = request.strip()
        if Path(text).exists():
            request = json.loads(Path(text).read_text(encoding="utf-8"))
        else:
            request = json.loads(text)
    if kwargs:
        request = {**request, **kwargs}

    action = str(request.get("action", request.get("op", "submit"))).lower()

    if action in ("ping", "health"):
        return {"ok": True, "status": "ok", "package": "operator_quant_groups"}

    if action in ("fixture", "load_fixture"):
        return {"ok": True, "fixture": _load_fixture(request.get("path"))}

    if action in ("eligibility", "qualify", "check"):
        payload = request.get("payload") or request.get("package") or _load_fixture(
            request.get("fixture_path")
        )
        report = evaluate_eligibility(
            payload,
            request.get("inputs") or payload.get("input"),
            request.get("strategy") or request.get("activation_strategy"),
            request.get("budget") or request.get("error_budget"),
            weight_version=request.get("weight_version"),
            profile=request.get("profile"),
            approx_bits=request.get("approx_bits"),
            chunk_size=request.get("chunk_size"),
            dtype=request.get("dtype"),
        )
        return {"ok": True, "eligibility": report.to_jsonable()}

    if action in ("dequantize", "dequant"):
        payload = request.get("payload") or _load_fixture(request.get("fixture_path"))
        op = GroupedQuantizedLinear(payload)
        return {
            "ok": True,
            "weights": [[float(v) for v in row] for row in op.dequantized_weights()],
        }

    if action in ("submit", "run", "execute", "evaluate", "linear"):
        payload = request.get("payload")
        if payload is None and request.get("fixture"):
            payload = _load_fixture(request.get("fixture_path"))
        inputs = (
            request.get("inputs")
            or request.get("x")
            or (payload.get("input") if isinstance(payload, dict) else None)
        )
        profile = request.get("profile", "ranking")
        result = _DEFAULT_SERVICE.submit(
            request.get("weight_version"),
            inputs,
            package=payload if isinstance(payload, dict) else None,
            profile=profile,
            budget=request.get("budget") or request.get("error_budget"),
            approx_bits=request.get("approx_bits"),
            chunk_size=request.get("chunk_size"),
            dtype=request.get("dtype"),
            precision=request.get("precision"),
            exact=request.get("exact"),
            trace_id=request.get("trace_id"),
        )
        eligibility = result.get("eligibility")
        return {
            "ok": True,
            "trace_id": result.get("trace_id"),
            "cache_hit": result.get("cache_hit"),
            "output": result.get("output"),
            "mode": result.get("mode"),
            "eligibility": eligibility.to_jsonable()
            if hasattr(eligibility, "to_jsonable")
            else eligibility,
            "fast_path_groups": result.get("fast_path_groups"),
            "fallback_groups": result.get("fallback_groups"),
        }

    if action in ("withdraw", "revoke"):
        version = request["weight_version"]
        outcome = _DEFAULT_SERVICE.withdraw(version, request.get("reason"))
        return {"ok": True, "withdrawal": outcome}

    if action in ("trace", "receipt"):
        trace_id = request["trace_id"]
        return {"ok": True, "trace": _DEFAULT_SERVICE.trace(trace_id)}

    return {"ok": False, "error": f"unknown action: {action!r}"}


# Module level convenience functions.
def submit_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
    if args and isinstance(args[0], dict):
        return handle_request(args[0], **kwargs)
    return handle_request(kwargs | {"action": "submit"})


run_request = submit_request
