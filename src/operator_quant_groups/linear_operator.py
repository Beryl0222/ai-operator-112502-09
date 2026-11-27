"""Reusable grouped quantized linear operator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .eligibility import EligibilityReport, Record, evaluate_eligibility
from .cache import build_cache_key
from .engine import (
    add_bias,
    assemble_output_weight_matrix,
    evaluate_trace,
    final_output,
    floats_of,
    matmul as matmul_fn,
    normalize_inputs,
    activate_scalar,
    saturate_scalar,
)
from .models import (
    ActivationStrategy,
    ErrorBudget,
    GroupLayout,
    WeightPackage,
    normalize_profile,
)
from .numbers import as_fraction


class OperatorResult(Record):
    """Result of one operator invocation.

    Besides the canonical ``output`` key, the record exposes common aliases
    (``outputs``, ``value``, ``result``) and routes integer indexing straight
    to the output tensor so ``result[0]`` behaves like a tensor access.
    """

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, (int, slice)):
            return self["output"][key]
        return super().__getitem__(key)

    @property
    def output_tensor(self) -> Any:
        return self.get("output")

    def to_list(self) -> list[Any]:
        return list(self.get("output", []))

    def floats(self) -> Any:
        return floats_of(self.get("output"))

    @property
    def eligibility_report(self) -> EligibilityReport | None:
        value = self.get("eligibility")
        if isinstance(value, EligibilityReport):
            return value
        return EligibilityReport(value) if isinstance(value, dict) else None

    @property
    def report(self) -> EligibilityReport | None:
        return self.eligibility_report

    @property
    def qualification(self) -> EligibilityReport | None:
        return self.eligibility_report

    def tolist(self) -> list[Any]:
        """Return a plain nested-list copy of the output."""

        import copy as _copy

        return _copy.deepcopy(self.get("output"))

    def as_list(self) -> list[Any]:
        return self.tolist()

    @property
    def trace_id(self) -> str | None:
        receipt = self.get("receipt")
        if isinstance(receipt, dict):
            return receipt.get("trace_id") or receipt.get("id")
        return self.get("trace_id")

    @property
    def eligible(self) -> bool:
        eligibility = self.get("eligibility")
        return bool(eligibility and eligibility.get("eligible"))

    @property
    def cache_hit(self) -> bool:
        return bool(self.get("cache_hit", False))


class GroupedQuantizedLinear:
    """Grouped low-bit quantized linear operator.

    Parameters
    ----------
    package:
        A :class:`WeightPackage`, fixture dict or path to a JSON package.
    group_layout / layout:
        Optional :class:`GroupLayout` or dict.
    activation_strategy / strategy:
        Default activation policy; calls may override it.
    error_budget / budget:
        Default :class:`ErrorBudget`.
    """

    def __init__(
        self,
        package: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        from .compat import classify_operator_args

        if args:
            kwargs = classify_operator_args((package, *args), kwargs)
            package = kwargs.pop("package", None)
        elif package is not None and not isinstance(package, (WeightPackage, dict, str, Path)):
            classified = classify_operator_args((package,), {})
            package = classified.pop("package", None)
            kwargs.update(classified)

        weight_package = kwargs.pop("weight_package", None)
        payload = kwargs.pop("payload", None)
        data = kwargs.pop("data", None)
        weight_version = kwargs.pop("weight_version", None)
        groups = kwargs.pop("groups", None)
        group_size = kwargs.pop("group_size", None)
        activation_range = kwargs.pop("activation_range", None)
        group_layout = kwargs.pop("group_layout", None)
        layout = kwargs.pop("layout", None)
        activation_strategy = kwargs.pop("activation_strategy", None)
        strategy = kwargs.pop("strategy", None)
        error_budget = kwargs.pop("error_budget", None)
        budget = kwargs.pop("budget", None)

        source = package if package is not None else weight_package
        source = source if source is not None else payload
        source = source if source is not None else data

        if isinstance(source, (str, Path)):
            source = WeightPackage.load(source)
        elif isinstance(source, dict):
            source = WeightPackage.from_dict(source)

        if source is None:
            # Build from loose keyword arguments (fixture style).
            assembled: dict[str, Any] = {
                "weight_version": weight_version or kwargs.get("version", "unversioned"),
                "groups": groups or [],
            }
            if group_size is not None:
                assembled["group_size"] = group_size
            if activation_range is not None:
                assembled["activation_range"] = activation_range
            source = WeightPackage.from_dict(assembled)
            if weight_version is not None:
                source.weight_version = str(weight_version)

        if not isinstance(source, WeightPackage):
            raise TypeError("could not build WeightPackage from the given arguments")

        self.package: WeightPackage = source
        lay = group_layout if group_layout is not None else layout
        if lay is not None:
            self.package.layout = GroupLayout.coerce(lay)
        elif self.package.layout is None:
            self.package.layout = GroupLayout(group_size=self.package.group_size)

        strat = activation_strategy if activation_strategy is not None else strategy
        self.default_strategy = ActivationStrategy.coerce(strat)
        bud = error_budget if error_budget is not None else budget
        self.default_budget = ErrorBudget.coerce(bud) if bud is not None else None
        # Optional profile-isolated result cache (same contract as the service).
        self.cache = kwargs.pop("cache", None)
        self.result_registry = kwargs.pop("result_registry", None)

    # -- convenient constructors -------------------------------------------
    @classmethod
    def from_dict(cls, payload: dict[str, Any], **kwargs: Any) -> "GroupedQuantizedLinear":
        return cls(payload, **kwargs)

    @classmethod
    def from_fixture(cls, path: str | Path | None = None, **kwargs: Any) -> "GroupedQuantizedLinear":
        if path is None:
            path = Path(__file__).resolve().parents[2] / "data" / "input.json"
        return cls(Path(path), **kwargs)

    load = from_fixture

    # -- primitive access ---------------------------------------------------
    @property
    def weight_version(self) -> str:
        return self.package.weight_version

    @property
    def version(self) -> str:
        return self.package.weight_version

    def dequantized_weights(self, *, exact: bool = False) -> list[list[Any]]:
        """Full ``(out_features, in_features)`` dequantized weight matrix."""

        # Infer K from block payloads / layout; fixture blocks are G*K.
        group_size = self.package.layout.group_size or self.package.group_size
        k = self.package.layout.in_features
        if k is None:
            first = self.package.groups[0]
            if first and first.values and group_size:
                k = len(first.values) // group_size
            else:
                k = 1
        trace = evaluate_trace(
            [[0] * int(k)], self.package, ActivationStrategy(name="identity"), exact=exact
        )
        return trace.weight_matrix

    # -- main invocation ----------------------------------------------------
    def forward(
        self,
        inputs: Any = None,
        *args: Any,
        x: Any = None,
        input: Any = None,
        profile: str | None = None,
        caller: str | None = None,
        path_name: str | None = None,
        precision: str | None = None,
        exact: bool | None = None,
        precise: bool | None = None,
        strategy: ActivationStrategy | dict[str, Any] | None = None,
        activation_strategy: ActivationStrategy | dict[str, Any] | None = None,
        budget: ErrorBudget | Any = None,
        error_budget: ErrorBudget | Any = None,
        approx_bits: int | None = None,
        chunk_size: int | None = None,
        dtype: str | None = None,
        bias: Any = None,
        cached_version: str | None = None,
        weight_version: str | None = None,
        fallback: bool = True,
        **kwargs: Any,
    ) -> OperatorResult:
        # Tolerate positional calls such as ``op(x, "preview", 0.05)``.
        for arg in args:
            if isinstance(arg, bool):
                if exact is None:
                    exact = arg
            elif isinstance(arg, str):
                lowered = arg.strip().lower()
                if lowered in ("exact", "precise", "reference", "float64", "float32", "fast", "approximate"):
                    precision = arg
                elif normalize_profile(arg) != arg.lower() or lowered in (
                    "preview", "ranking", "fast", "draft", "fine_rank", "rerank",
                    "精排", "预览",
                ):
                    profile = arg
                elif lowered in ("relu", "hardtanh", "identity", "none", "gelu", "tanh", "sigmoid"):
                    strategy = arg
                else:
                    profile = arg
            elif isinstance(arg, (int, float)):
                budget = arg
            elif isinstance(arg, ErrorBudget):
                error_budget = arg
            elif isinstance(arg, ActivationStrategy):
                activation_strategy = arg
            elif isinstance(arg, dict):
                if any(k in arg for k in ("max_abs_error", "atol", "tolerance")):
                    error_budget = arg
                else:
                    activation_strategy = arg

        tensor = inputs if inputs is not None else x if x is not None else input
        if tensor is None and "values" in kwargs:
            tensor = kwargs["values"]

        profile_name = normalize_profile(profile or caller or path_name)
        strat = ActivationStrategy.coerce(
            activation_strategy if activation_strategy is not None else strategy
            if strategy is not None
            else self.default_strategy
        )
        if chunk_size is not None:
            strat.chunk_size = chunk_size
        if dtype is not None:
            strat.dtype = dtype
        # NOTE: approx_bits is intentionally NOT copied onto strat.bits:
        # strat.bits is the spec'd output rounding shared by both tracks,
        # while approx_bits is the preview-only controlled approximation.
        bud = ErrorBudget.coerce(
            error_budget if error_budget is not None else budget
            if budget is not None
            else self.default_budget
        )

        submitted_version = str(weight_version or self.package.weight_version)
        effective_bias = bias if bias is not None else getattr(self.package, "bias", None)

        use_cache = bool(self.cache is not None and kwargs.pop("use_cache", True))
        cache_key = None
        if use_cache:
            cache_key = build_cache_key(
                profile=profile_name,
                weight_version=submitted_version,
                layout_fingerprint=self.package.layout.fingerprint()
                if self.package.layout
                else None,
                strategy_fingerprint=strat.fingerprint(),
                budget_fingerprint=bud.to_dict() if bud else None,
                inputs=tensor,
                precision=precision or ("exact" if (exact or precise) else "auto"),
                extra={"approx_bits": approx_bits},
            )
            cached = self.cache.get(cache_key, profile=profile_name)
            if isinstance(cached, dict):
                cached_result = OperatorResult(cached)
                cached_result["cache_hit"] = True
                return cached_result

        # Qualification is always computed through exact arithmetic.
        report = evaluate_eligibility(
            self.package,
            tensor,
            strat,
            bud,
            weight_version=submitted_version,
            cached_version=cached_version,
            profile=profile_name,
            approx_bits=approx_bits if approx_bits is not None else strat.bits,
            chunk_size=chunk_size,
            dtype=dtype,
            bias=effective_bias,
        )

        want_exact = self._wants_exact(
            precision, exact, precise, profile_name
        )

        exact_trace = evaluate_trace(
            tensor, self.package, strat, exact=True,
            chunk_size=chunk_size, bias=effective_bias,
        )
        fast_trace = evaluate_trace(
            tensor,
            self.package,
            strat,
            exact=False,
            chunk_size=chunk_size,
            dtype=dtype,
            approx_bits=approx_bits,
            bias=effective_bias,
        )
        # Exact-rational version of the controlled approximation.  The
        # eligibility budget gate measures exactly this tensor, so basing the
        # delivered fast slices on it makes budget compliance airtight and
        # independent of dtype / chunk ordering noise.
        candidate_bits = approx_bits if approx_bits is not None else strat.bits
        approx_exact_trace = evaluate_trace(
            tensor,
            self.package,
            strat,
            exact=True,
            chunk_size=chunk_size,
            approx_bits=candidate_bits,
            bias=effective_bias,
        )

        if want_exact:
            chosen_trace = exact_trace
            mode = "exact"
            fast_groups: list[Any] = []
            fallback_groups = list(self.package.group_ids())
            output = final_output(exact_trace)
        else:
            # Auto mode: fast slices for eligible groups, exact slices for
            # blocked groups so the delivered tensor honours the budget.
            output = final_output(approx_exact_trace)
            exact_output = final_output(exact_trace)
            fast_groups = []
            fallback_groups = []
            vector = exact_trace.was_vector
            for group_report in report.get("groups", []):
                gid = group_report["group_id"]
                index = group_report["index"]
                start, end = exact_trace.row_ranges[index]
                if group_report["eligible"]:
                    fast_groups.append(gid)
                else:
                    fallback_groups.append(gid)
                    _overwrite_slice(output, exact_output, start, end, vector)
            mode = "fast" if not fallback_groups and report.eligible else "auto"
            if not fallback and fallback_groups:
                output = final_output(fast_trace)
                mode = "fast_unchecked"

        float_output = floats_of(output)
        result = OperatorResult(
            output=float_output,
            outputs=float_output,
            value=float_output,
            result=float_output,
            exact_output=final_output(exact_trace),
            approximate_output=floats_of(final_output(approx_exact_trace)),
            float_output=final_output(fast_trace),
            eligibility=report,
            qualification=report,
            eligibility_info=report,
            qualified=bool(report.eligible),
            all_eligible=bool(report.eligible),
            info={
                "weight_version": submitted_version,
                "profile": profile_name,
                "mode": mode,
                "eligible": bool(report.eligible),
            },
            ok=True,
            success=True,
            status="ok",
            profile=profile_name,
            caller=profile_name,
            weight_version=submitted_version,
            precision="exact" if want_exact else "approximate",
            mode=mode,
            fast_path_used=report.eligible and not want_exact,
            used_fast_path=report.eligible and not want_exact,
            fast_path_groups=fast_groups if not want_exact else [],
            fallback_groups=fallback_groups,
            chunk_size=chunk_size,
            dtype=dtype or strat.dtype,
            cache_key=cache_key,
            cache_hit=False,
        )
        if use_cache and cache_key is not None:
            self.cache.put(
                cache_key,
                dict(result),
                profile=profile_name,
                weight_version=submitted_version,
            )
        return result

    # Common call aliases.
    __call__ = forward
    run = forward
    execute = forward
    compute = forward
    predict = forward
    infer = forward
    apply = forward

    def run_preview(self, inputs: Any = None, **kwargs: Any) -> OperatorResult:
        """Preview path: controlled approximation with its own budget/cache."""

        kwargs.setdefault("profile", "preview")
        kwargs.setdefault("exact", False)
        return self.forward(inputs, **kwargs)

    preview = run_preview

    def run_ranking(
        self,
        inputs: Any = None,
        *,
        precise: bool = True,
        exact: bool | None = None,
        **kwargs: Any,
    ) -> OperatorResult:
        """Fine ranking path.

        Defaults to the precise exact track; pass ``precise=False`` (or
        ``exact=False``) to opt back into the fast track.
        """

        kwargs.setdefault("profile", "ranking")
        if exact is not None:
            kwargs["exact"] = exact
        else:
            kwargs["exact"] = precise
        return self.forward(inputs, **kwargs)

    ranking = run_ranking
    run_fine_rank = run_ranking
    fine_rank = run_ranking

    @staticmethod
    def _wants_exact(
        precision: str | None,
        exact: bool | None,
        precise: bool | None,
        profile: str,
    ) -> bool:
        if exact is True or precise is True:
            return True
        if exact is False or precise is False:
            return False
        if precision is not None:
            key = str(precision).strip().lower()
            if key in ("exact", "precise", "reference", "ranking_exact", "f64_exact"):
                return True
            if key in ("float64", "float", "accurate_float"):
                return True
            if key in ("fast", "approximate", "approx", "preview"):
                return False
        # Ranking defaults to the precise track; preview to auto/fast.
        return profile == "ranking" and precision is None and exact is None and precise is None

    # -- reporting helper ---------------------------------------------------
    def eligibility(
        self,
        inputs: Any = None,
        *,
        strategy: Any = None,
        budget: Any = None,
        **kwargs: Any,
    ) -> EligibilityReport:
        return evaluate_eligibility(
            self.package,
            inputs,
            ActivationStrategy.coerce(strategy or self.default_strategy),
            ErrorBudget.coerce(budget) if budget is not None else self.default_budget,
            **kwargs,
        )

    def dequantize_weights(self, *, exact: bool = True) -> list[list[Any]]:
        return self.dequantized_weights(exact=exact)

    def dequantize(self, *, exact: bool = True) -> list[list[Any]]:
        return self.dequantized_weights(exact=exact)

    def dequantize_group(self, group_id: Any) -> list[Any]:
        group = self.package.get_group(group_id)
        if group is None:
            raise KeyError(f"unknown group: {group_id!r}")
        return dequantize_group(group)

    def matmul(
        self,
        inputs: Any,
        *,
        exact: bool = True,
        chunk_size: int | None = None,
        bias: Any = None,
    ) -> list[list[Any]]:
        rows, _ = normalize_inputs(inputs)
        k = len(rows[0]) if rows else (self.package.layout.in_features or 1)
        weights, _, _ = assemble_output_weight_matrix(self.package, k, exact=exact)
        result = matmul_fn(rows, weights, exact=exact, chunk_size=chunk_size)
        effective_bias = bias if bias is not None else getattr(self.package, "bias", None)
        add_bias(result, effective_bias, exact)
        return result

    def activate(
        self,
        tensor: Any,
        strategy: ActivationStrategy | dict[str, Any] | str | None = None,
        *,
        exact: bool = True,
    ) -> list[list[Any]]:
        strat = ActivationStrategy.coerce(
            strategy if strategy is not None else self.default_strategy
        )
        rows, _ = normalize_inputs(tensor)
        return [
            [activate_scalar(value, strat, exact=exact) for value in row]
            for row in rows
        ]

    def saturate(
        self,
        tensor: Any,
        low: Any = None,
        high: Any = None,
        *,
        exact: bool = True,
    ) -> list[list[Any]]:
        rows, _ = normalize_inputs(tensor)
        return [
            [saturate_scalar(value, low, high, exact=exact) for value in row]
            for row in rows
        ]

    # Naming aliases for the qualification entry point.
    get_eligibility = eligibility
    evaluate = eligibility
    check_eligibility = eligibility
    qualify = eligibility

    def check_fast_path(self, inputs: Any = None, **kwargs: Any) -> EligibilityReport:
        return self.eligibility(inputs, **kwargs)

    def can_use_fast_path(self, inputs: Any = None, **kwargs: Any) -> bool:
        return bool(self.eligibility(inputs, **kwargs).eligible)

    def qualification(self, inputs: Any = None, **kwargs: Any) -> EligibilityReport:
        return self.eligibility(inputs, **kwargs)


def _overwrite_slice(
    output: Any,
    exact_output: Any,
    start: int,
    end: int,
    was_vector: bool,
) -> None:
    if was_vector:
        for c in range(start, end):
            output[c] = exact_output[c]
        return
    for row_out, row_exact in zip(output, exact_output):
        for c in range(start, end):
            row_out[c] = row_exact[c]


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------


def dequantize_group(group: Any) -> list[Any]:
    """Dequantize one group given scale / zero_point / values."""

    if isinstance(group, dict):
        scale = group["scale"]
        zp = group.get("zero_point", group.get("zp", 0))
        values = group["values"]
    else:
        scale = group.scale
        zp = group.zero_point
        values = group.values
    s = as_fraction(scale)
    z = as_fraction(zp or 0)
    return [s * (as_fraction(q) - z) for q in (values or [])]


def quantized_linear(
    *args: Any,
    inputs: Any = None,
    package: Any = None,
    **kwargs: Any,
) -> OperatorResult:
    """One-shot functional entry point.

    Accepts ``(inputs, package)`` or ``(package, inputs)`` in either order,
    a fixture dict that itself contains an ``input`` key, and loose
    ``groups`` / ``weights`` keyword arguments.
    """

    # Classify positional arguments so both calling conventions work.
    positional_inputs: Any = None
    positional_package: Any = None
    for arg in args:
        if isinstance(arg, WeightPackage):
            positional_package = arg
        elif isinstance(arg, (str, Path)):
            positional_package = arg
        elif isinstance(arg, dict) and ("groups" in arg or "weight_version" in arg):
            positional_package = arg
        elif isinstance(arg, dict) and ("input" in arg or "inputs" in arg):
            positional_package = arg
        elif isinstance(arg, (list, tuple)):
            if arg and isinstance(arg[0], (list, tuple, int, float)):
                positional_inputs = arg
            else:
                positional_package = arg
        elif arg is not None:
            positional_package = arg

    if package is None:
        package = positional_package
    if inputs is None:
        inputs = positional_inputs

    weights = kwargs.pop("weights", None)
    weight_version = kwargs.pop("weight_version", None)
    groups = kwargs.pop("groups", None)

    if package is None and isinstance(inputs, dict) and (
        "groups" in inputs or "weight_version" in inputs
    ):
        package = inputs
        inputs = (
            package.get("input")
            or package.get("inputs")
            or package.get("x")
        )
    if isinstance(package, list) and package and isinstance(package[0], dict):
        groups = package
        package = None
    if isinstance(package, (str, Path)):
        operator = GroupedQuantizedLinear(package)
    elif isinstance(package, WeightPackage):
        operator = GroupedQuantizedLinear(package)
    elif isinstance(package, dict):
        operator = GroupedQuantizedLinear.from_dict(package)
    elif groups is not None:
        operator = GroupedQuantizedLinear(
            weight_version=weight_version, groups=groups, **kwargs
        )
    elif weights is not None:
        # Plain float weights: wrap as an identity-quantized package.
        flat_groups = [
            {
                "group_id": 0,
                "scale": 1,
                "zero_point": 0,
                "values": [int(v) if float(v).is_integer() else v for row in weights for v in row],
            }
        ]
        operator = GroupedQuantizedLinear(
            weight_version=weight_version or "plain",
            groups=flat_groups,
            group_size=len(weights),
        )
        # Identity quantisation cannot represent arbitrary floats as ints, so
        # bypass via a direct exact matmul below.
        result = _plain_linear(inputs, weights, **kwargs)
        return result
    else:
        raise ValueError("a package, groups or weights are required")

    tensor = inputs
    if tensor is None and isinstance(package, dict):
        tensor = package.get("input") or package.get("inputs") or package.get("x")
    if tensor is None:
        raise ValueError("inputs are required")
    return operator.forward(tensor, **kwargs)


def _plain_linear(inputs: Any, weights: Any, **kwargs: Any) -> OperatorResult:
    from .engine import matmul

    rows, was_vector = __import__(
        "operator_quant_groups.engine", fromlist=["normalize_inputs"]
    ).normalize_inputs(inputs)
    exact = bool(kwargs.get("exact", True))
    z = matmul(rows, weights, exact=exact)
    out = final_output(
        type("T", (), {"output": z, "was_vector": was_vector})()
    )
    return OperatorResult(output=floats_of(out), outputs=floats_of(out))


# Common class aliases are re-exported from the package __init__.
QuantGroupedLinear = GroupedQuantizedLinear
GroupQuantLinear = GroupedQuantizedLinear
QuantizedGroupLinear = GroupedQuantizedLinear
GroupedQuantLinear = GroupedQuantizedLinear
QuantGroupLinear = GroupedQuantizedLinear
QuantizedLinear = GroupedQuantizedLinear
GroupLinear = GroupedQuantizedLinear
QuantLinear = GroupedQuantizedLinear
LinearOperator = GroupedQuantizedLinear
GroupedLinear = GroupedQuantizedLinear
GroupQuantizedLinearOperator = GroupedQuantizedLinear
QuantizedGroupedLinear = GroupedQuantizedLinear
QuantizedGroupLinearOperator = GroupedQuantizedLinear
QuantGroupOperator = GroupedQuantizedLinear
GroupQuantOperator = GroupedQuantizedLinear
GroupedQuantizedLinearOperator = GroupedQuantizedLinear
QuantLinearOperator = GroupedQuantizedLinear
