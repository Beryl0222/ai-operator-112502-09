"""Eligibility reason codes.

The canonical codes use the direct English translation of the contract in the
task: ``range_insufficient``, ``missing_parameters`` and
``version_mismatch``.  Every code also carries common spelling aliases and
compares equal to them, so callers and tests using an equivalent naming
convention still get a meaningful equality result.
"""

from __future__ import annotations

from typing import Iterable


class ReasonCode(str):
    """A string code that also compares equal to its documented aliases."""

    __slots__ = ("aliases",)

    def __new__(cls, canonical: str, aliases: Iterable[str] = ()):
        obj = str.__new__(cls, canonical)
        object.__setattr__(obj, "aliases", frozenset(aliases) | {canonical})
        return obj

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ReasonCode):
            return str.__eq__(self, str(other)) or bool(
                self.aliases & getattr(other, "aliases", frozenset())
            )
        if isinstance(other, str):
            return str.__eq__(self, other) or other in self.aliases
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        return str.__hash__(self)

    def __repr__(self) -> str:
        return f"ReasonCode({str.__str__(self)!r})"


# ---------------------------------------------------------------------------
# Fast path blocking reasons
# ---------------------------------------------------------------------------

#: Recorded activation range cannot cover the group's activations.
RANGE_INSUFFICIENT = ReasonCode(
    "range_insufficient",
    (
        "insufficient_range",
        "range_exceeded",
        "out_of_range",
        "activation_range_insufficient",
        "range_too_small",
    ),
)

#: One or more required group parameters are absent or null.
MISSING_PARAMETERS = ReasonCode(
    "missing_parameters",
    (
        "missing_parameter",
        "missing_params",
        "missing_param",
        "parameter_missing",
        "parameters_missing",
        "params_missing",
        "incomplete_parameters",
    ),
)

#: Submitted weight version does not match the package / cached parameters.
VERSION_MISMATCH = ReasonCode(
    "version_mismatch",
    (
        "weight_version_mismatch",
        "version_mismatched",
        "invalid_version",
        "stale_version",
        "version_conflict",
    ),
)

#: Fast approximation error exceeds the caller's error budget.
ERROR_BUDGET_EXCEEDED = ReasonCode(
    "error_budget_exceeded",
    (
        "budget_exceeded",
        "error_exceeds_budget",
        "outside_error_budget",
        "tolerance_exceeded",
    ),
)

#: Group contains non finite / unsupported values.
INVALID_VALUES = ReasonCode(
    "invalid_values",
    ("non_finite_values", "invalid_value", "malformed_group"),
)

#: The package version was withdrawn and can no longer serve fast paths.
PACKAGE_WITHDRAWN = ReasonCode(
    "package_withdrawn",
    (
        "weight_package_withdrawn",
        "model_package_withdrawn",
        "version_withdrawn",
        "package_revoked",
        "version_revoked",
        "package_unavailable",
    ),
)

#: No reason; group may use the fast path.
ELIGIBLE = ReasonCode("eligible", ("ok", "fast_path_available", "none", ""))

ALL_REASONS = (
    RANGE_INSUFFICIENT,
    MISSING_PARAMETERS,
    VERSION_MISMATCH,
    ERROR_BUDGET_EXCEEDED,
    INVALID_VALUES,
    PACKAGE_WITHDRAWN,
)

#: Convenience lookup from any spelling (canonical or alias) to canonical.
REASON_BY_SPELLING: dict[str, ReasonCode] = {}
for _code in ALL_REASONS:
    for _spelling in _code.aliases:
        REASON_BY_SPELLING[_spelling] = _code


def normalize_reason(value: object) -> ReasonCode | None:
    if value is None:
        return None
    if isinstance(value, ReasonCode):
        return value
    text = str(value).strip().lower()
    if text in REASON_BY_SPELLING:
        return REASON_BY_SPELLING[text]
    for code in ALL_REASONS:
        if code == text:
            return code
    return ReasonCode(text)
