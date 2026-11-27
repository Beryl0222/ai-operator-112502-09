"""Eligibility / qualification reports for the grouped fast path."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

from . import reasons as reason_codes
from .engine import evaluate_trace
from .models import (
    ActivationStrategy,
    ErrorBudget,
    WeightPackage,
)
from .numbers import as_fraction, overflow_distance


class Record(dict):
    """Dict record with attribute access and JSON friendly conversion."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def copy_record(self) -> "Record":
        return Record(_deep_jsonable(self))


def _deep_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _deep_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_jsonable(v) for v in value]
    if isinstance(value, Fraction):
        return float(value)
    return value


# ---------------------------------------------------------------------------
# Group qualification
# ---------------------------------------------------------------------------


class GroupEligibility(Record):
    @property
    def reason(self) -> reason_codes.ReasonCode | None:
        value = self.get("reason_code")
        return reason_codes.normalize_reason(value) if value is not None else None

    @property
    def primary_reason(self) -> reason_codes.ReasonCode | None:
        return self.reason

    def has_reason(self, code: Any) -> bool:
        target = reason_codes.normalize_reason(code)
        return any(reason_codes.normalize_reason(r) == target for r in self.get("reasons", []))

    # -- boolean aliases ----------------------------------------------------
    @property
    def range_insufficient(self) -> bool:
        return self.has_reason(reason_codes.RANGE_INSUFFICIENT)

    @property
    def insufficient_range(self) -> bool:
        return self.range_insufficient

    @property
    def range_exceeded(self) -> bool:
        return self.range_insufficient

    @property
    def out_of_range(self) -> bool:
        return self.range_insufficient

    @property
    def missing_parameters(self) -> bool:
        return self.has_reason(reason_codes.MISSING_PARAMETERS)

    @property
    def missing_parameter(self) -> bool:
        return self.missing_parameters

    @property
    def params_missing(self) -> bool:
        return self.missing_parameters

    @property
    def version_mismatch(self) -> bool:
        return self.has_reason(reason_codes.VERSION_MISMATCH)

    @property
    def error_budget_exceeded(self) -> bool:
        return self.has_reason(reason_codes.ERROR_BUDGET_EXCEEDED)

    @property
    def eligible(self) -> bool:
        return bool(self.get("eligible", False))

    @property
    def fast_path_eligible(self) -> bool:
        return self.eligible

    @property
    def can_use_fast_path(self) -> bool:
        return self.eligible


class EligibilityReport(Record):
    """Aggregate qualification for every group."""

    @property
    def eligible(self) -> bool:
        return bool(self.get("eligible", False))

    @property
    def fast_path_eligible(self) -> bool:
        return self.eligible

    @property
    def can_use_fast_path(self) -> bool:
        return self.eligible

    @property
    def all_eligible(self) -> bool:
        return self.eligible

    def group(self, group_id: Any) -> GroupEligibility | None:
        for item in self.get("groups", []):
            if item.get("group_id") == group_id or item.get("index") == group_id:
                return item if isinstance(item, GroupEligibility) else GroupEligibility(item)
        return None

    def reasons_for(self, group_id: Any) -> list[str]:
        item = self.group(group_id)
        return list(item["reasons"]) if item else []

    @property
    def range_insufficient_groups(self) -> list[Any]:
        return self.get("by_reason", {}).get(reason_codes.RANGE_INSUFFICIENT, [])

    @property
    def missing_parameter_groups(self) -> list[Any]:
        return self.get("by_reason", {}).get(reason_codes.MISSING_PARAMETERS, [])

    @property
    def version_mismatch_groups(self) -> list[Any]:
        return self.get("by_reason", {}).get(reason_codes.VERSION_MISMATCH, [])

    @property
    def blocked_groups(self) -> list[Any]:
        return self.get("ineligible_groups", [])

    @property
    def ineligible_group_ids(self) -> list[Any]:
        return self.get("ineligible_groups", [])

    def summary(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "eligible_groups": list(self.get("eligible_groups", [])),
            "ineligible_groups": list(self.get("ineligible_groups", [])),
            "by_reason": {
                str(code): list(ids) for code, ids in self.get("by_reason", {}).items()
            },
        }

    def to_jsonable(self) -> dict[str, Any]:
        return _deep_jsonable(dict(self))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

#: Stable precedence when a group has several independent defects.
REASON_PRECEDENCE = (
    reason_codes.VERSION_MISMATCH,
    reason_codes.MISSING_PARAMETERS,
    reason_codes.INVALID_VALUES,
    reason_codes.RANGE_INSUFFICIENT,
    reason_codes.ERROR_BUDGET_EXCEEDED,
)


def _primary(reasons: list[reason_codes.ReasonCode]) -> reason_codes.ReasonCode | None:
    for code in REASON_PRECEDENCE:
        for given in reasons:
            if reason_codes.normalize_reason(given) == code:
                return code
    return reasons[0] if reasons else None


def evaluate_eligibility(
    package: WeightPackage | dict[str, Any] | None = None,
    inputs: Any = None,
    strategy: ActivationStrategy | dict[str, Any] | None = None,
    budget: ErrorBudget | Any = None,
    *,
    weight_version: str | None = None,
    expected_version: str | None = None,
    cached_version: str | None = None,
    actual_version: str | None = None,
    profile: str | None = None,
    approx_bits: int | None = None,
    chunk_size: int | None = None,
    dtype: str | None = None,
    bias: Any = None,
    require_activation_range: bool = True,
) -> EligibilityReport:
    """Compute per-group fast path qualification.

    The decision is deterministic: exact rational arithmetic is used for range
    and budget boundaries, while *dtype* / *chunk_size* only shape optional
    float diagnostics.
    """

    if isinstance(package, dict):
        package = WeightPackage.from_dict(package)
    if package is None:
        raise ValueError("package is required for eligibility evaluation")
    if not isinstance(package, WeightPackage):
        # Permit lightweight duck packages.
        package = WeightPackage.from_dict(dict(package))  # type: ignore[arg-type]

    strat = ActivationStrategy.coerce(strategy)
    if chunk_size is not None:
        strat.chunk_size = chunk_size
    if dtype is not None:
        strat.dtype = dtype
    bud = ErrorBudget.coerce(budget)

    submitted_version = weight_version or expected_version or package.weight_version
    cached_actual = cached_version if cached_version is not None else actual_version

    # Exact reference trace (no lossy approximation unless explicitly asked;
    # the budget gate uses the exact approximation trace below).  Inputs are
    # optional: without them only structural checks (version / parameters)
    # are produced.
    if inputs is not None:
        reference = evaluate_trace(
            inputs,
            package,
            strat,
            exact=True,
            chunk_size=chunk_size,
            bias=bias,
        )
        approximate_exact = evaluate_trace(
            inputs,
            package,
            strat,
            exact=True,
            chunk_size=chunk_size,
            approx_bits=approx_bits,
            bias=bias,
        )
    else:
        reference = None
        approximate_exact = None

    group_reports: list[GroupEligibility] = []
    eligible_ids: list[Any] = []
    ineligible_ids: list[Any] = []
    by_reason: dict[str, list[Any]] = {
        str(code): [] for code in reason_codes.ALL_REASONS
    }

    for index, group in enumerate(package.groups):
        gid = group.group_id
        reasons: list[reason_codes.ReasonCode] = []
        missing_fields: list[str] = []
        expected_v = str(submitted_version) if submitted_version is not None else None
        actual_v: str | None = None

        # 1) Version consistency ------------------------------------------------
        group_version = group.weight_version or package.weight_version
        version_source = "group" if group.weight_version else "package"
        if cached_actual is not None and str(cached_actual) != expected_v:
            reasons.append(reason_codes.VERSION_MISMATCH)
            actual_v = str(cached_actual)
        elif group_version is not None and expected_v is not None and str(group_version) != expected_v:
            reasons.append(reason_codes.VERSION_MISMATCH)
            actual_v = str(group_version)
            version_source = version_source
        else:
            actual_v = str(group_version) if group_version is not None else None

        # 2) Missing parameters --------------------------------------------------
        # Range resolution mirrors the engine: an explicit strategy range
        # wins; otherwise use the group range with the package default.
        strategy_range = strat.resolved_range()
        if strategy_range is not None:
            resolved_range = strategy_range
            range_source = "strategy"
        else:
            default_range = package.activation_range
            resolved_range = group.resolved_activation_range(default_range)
            range_source = "group" if group.activation_range is not None else "package"
        missing_range = require_activation_range and resolved_range is None
        missing_fields = group.missing_fields(require_activation_range=missing_range)
        if missing_fields:
            reasons.append(reason_codes.MISSING_PARAMETERS)

        # 3) Value validity ------------------------------------------------------
        invalid = False
        if group.values is not None:
            for q in group.values:
                try:
                    as_fraction(q)
                except (TypeError, ValueError):
                    invalid = True
                    break
        if invalid:
            reasons.append(reason_codes.INVALID_VALUES)

        if reference is not None and index < len(reference.row_ranges):
            start, end = reference.row_ranges[index]
        else:
            start, end = (0, 0)

        # Metrics are populated when computation is possible.
        observed_low: Fraction | None = None
        observed_high: Fraction | None = None
        max_overflow = Fraction(0)
        boundary_state: str | None = None
        max_abs_error = Fraction(0)
        mean_abs = Fraction(0)
        within_budget = True

        computable = (
            inputs is not None
            and reference is not None
            and not missing_fields
            and not invalid
            and index < len(reference.row_ranges)
        )

        if computable:
            # 4) Range adequacy against pre-saturation activation values.
            if resolved_range is not None:
                lo = as_fraction(resolved_range[0])
                hi = as_fraction(resolved_range[1])
                probe = reference.group_tensor("activated", (start, end))
                for row in probe:
                    for value in row:
                        v = as_fraction(value)
                        if observed_low is None or v < observed_low:
                            observed_low = v
                        if observed_high is None or v > observed_high:
                            observed_high = v
                        overflow = overflow_distance(v, lo, hi)
                        if overflow > max_overflow:
                            max_overflow = overflow
                if max_overflow > 0:
                    # Default ("strict_range") treats any uncovered activation
                    # as an insufficient range.  With strict_range=False the
                    # caller allows saturation within the error budget, so a
                    # small overflow is tolerated when the budget covers it.
                    tolerance = (
                        as_fraction(bud.max_abs_error)
                        if not bud.strict_range and bud.max_abs_error is not None
                        else Fraction(0)
                    )
                    if max_overflow > tolerance:
                        reasons.append(reason_codes.RANGE_INSUFFICIENT)
                if observed_low is not None and observed_high is not None:
                    if observed_low < lo:
                        boundary_state = "below_range"
                    elif observed_high > hi:
                        boundary_state = "above_range"
                    elif observed_low == lo or observed_high == hi:
                        boundary_state = "at_bound"
                    else:
                        boundary_state = "inside_range"
            elif require_activation_range:
                # Should already be in missing_fields, but stay defensive.
                if reason_codes.MISSING_PARAMETERS not in reasons:
                    reasons.append(reason_codes.MISSING_PARAMETERS)
                if "activation_range" not in missing_fields:
                    missing_fields.append("activation_range")

            # 5) Error budget against the exact approximation track.
            ref_slice = reference.group_tensor("output", (start, end))
            fast_slice = approximate_exact.group_tensor("output", (start, end))
            max_abs_error = _max_abs(ref_slice, fast_slice)
            mean_abs = _mean_abs(ref_slice, fast_slice)
            within_budget = bud.allows(max_abs_error, mean_abs)
            if not within_budget:
                reasons.append(reason_codes.ERROR_BUDGET_EXCEEDED)

        is_eligible = not reasons
        primary = _primary(reasons)

        record = GroupEligibility(
            group_id=gid,
            index=index,
            eligible=is_eligible,
            fast_path_eligible=is_eligible,
            can_use_fast_path=is_eligible,
            reasons=[str(r) for r in reasons],
            reason_code=str(primary) if primary is not None else None,
            reason=str(primary) if primary is not None else None,
            primary_reason=str(primary) if primary is not None else None,
            missing_fields=missing_fields,
            expected_version=expected_v,
            actual_version=actual_v,
            cached_version=str(cached_actual) if cached_actual is not None else None,
            version_source=version_source,
            range={
                "source": range_source if computable or resolved_range is not None else None,
                "low": _num(resolved_range[0]) if resolved_range else None,
                "high": _num(resolved_range[1]) if resolved_range else None,
                "observed_low": _num(observed_low),
                "observed_high": _num(observed_high),
                "max_overflow": _num(max_overflow),
                "boundary_state": boundary_state,
                "sufficient": reason_codes.RANGE_INSUFFICIENT not in reasons,
            },
            error={
                "max_abs_error": _num(max_abs_error),
                "mean_abs_error": _num(mean_abs),
                "budget": bud.to_dict(),
                "within_budget": within_budget,
                "approx_bits": approx_bits,
            },
            dtype=dtype or strat.dtype,
            chunk_size=chunk_size or strat.chunk_size,
        )
        group_reports.append(record)

        if is_eligible:
            eligible_ids.append(gid)
        else:
            ineligible_ids.append(gid)
            seen: set[str] = set()
            for code in reasons:
                canonical = str(reason_codes.normalize_reason(code))
                if canonical not in seen:
                    by_reason[canonical].append(gid)
                    seen.add(canonical)

    overall_eligible = len(ineligible_ids) == 0 and len(package.groups) > 0
    report = EligibilityReport(
        eligible=overall_eligible,
        fast_path_eligible=overall_eligible,
        can_use_fast_path=overall_eligible,
        weight_version=submitted_version,
        profile=profile,
        groups=group_reports,
        eligible_groups=eligible_ids,
        ineligible_groups=ineligible_ids,
        blocked_groups=ineligible_ids,
        by_reason=by_reason,
        total_groups=len(package.groups),
        eligible_count=len(eligible_ids),
        ineligible_count=len(ineligible_ids),
        budget=bud.to_dict(),
        strategy=strat.to_dict(),
    )
    return report


def _num(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Fraction):
        return float(value)
    return value


class ErrorHandoff(Record):
    """Error budget handed between the dequant / linear / activation stages.

    The handoff records the exact measured error and whether the receiving
    path's budget accepts it, giving callers an auditable "error hand-off".
    """

    @property
    def within_budget(self) -> bool:
        return bool(self.get("within_budget", True))

    @property
    def accepted(self) -> bool:
        return self.within_budget

    @property
    def max_abs_error(self) -> float:
        return self.get("max_abs_error", 0.0)


def handoff_error(
    reference: Any,
    candidate: Any,
    budget: Any = None,
    *,
    stage: str = "output",
    chunk_size: int | None = None,
) -> ErrorHandoff:
    """Measure ``candidate`` against ``reference`` and compare to ``budget``."""

    from .numbers import max_abs_error as _max_abs
    from .numbers import mean_abs_error as _mean_abs

    max_abs = _max_abs(candidate, reference, chunk_size=chunk_size)
    mean_abs = _mean_abs(candidate, reference)
    bud = ErrorBudget.coerce(budget) if budget is not None else ErrorBudget()
    within = bud.allows(max_abs, mean_abs)
    return ErrorHandoff(
        stage=stage,
        max_abs_error=float(max_abs),
        max_abs_error_exact=str(max_abs),
        mean_abs_error=float(mean_abs),
        budget=bud.to_dict(),
        within_budget=within,
        accepted=within,
    )


def _flatten_list(value: Any) -> list[Any]:
    out: list[Any] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_flatten_list(item))
    else:
        out.append(value)
    return out


def _max_abs(a: list[list[Any]], b: list[list[Any]]) -> Fraction:
    worst = Fraction(0)
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            d = abs(as_fraction(x) - as_fraction(y))
            if d > worst:
                worst = d
    return worst


def _mean_abs(a: list[list[Any]], b: list[list[Any]]) -> Fraction:
    total = Fraction(0)
    count = 0
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            total += abs(as_fraction(x) - as_fraction(y))
            count += 1
    return total / count if count else Fraction(0)
