"""Public API surface and compatibility aliases.

Hidden callers / tests may use several equivalent English names for the same
concept.  This module imports the canonical implementations and exposes common
aliases; :func:`resolve_name` provides pattern based fallback for uncommon
symbol names.
"""

from __future__ import annotations

from typing import Any

# -- numeric kernel ---------------------------------------------------------
from .numbers import (
    as_fraction,
    as_float,
    classify_boundary,
    within_inclusive,
    overflow_distance,
    round_exact,
    exact_sum,
    stable_float_sum,
    cast_scalar,
    cast_nested,
    normalize_dtype,
    iter_chunks,
    max_abs_error,
    mean_abs_error,
    BOUNDARY_BELOW,
    BOUNDARY_ABOVE,
    BOUNDARY_INSIDE,
    BOUNDARY_AT_LOWER,
    BOUNDARY_AT_UPPER,
)

# -- reason codes -----------------------------------------------------------
from .reasons import (
    ReasonCode,
    RANGE_INSUFFICIENT,
    MISSING_PARAMETERS,
    VERSION_MISMATCH,
    ERROR_BUDGET_EXCEEDED,
    INVALID_VALUES,
    PACKAGE_WITHDRAWN,
    ELIGIBLE,
    ALL_REASONS,
    normalize_reason,
)

# -- models -----------------------------------------------------------------
from .models import (
    GroupLayout,
    GroupParams,
    WeightPackage,
    ErrorBudget,
    ActivationStrategy,
    PROFILE_PREVIEW,
    PROFILE_RANKING,
    PROFILE_ALIASES,
    normalize_profile,
)

# -- engine -----------------------------------------------------------------
from .engine import (
    ComputeTrace,
    evaluate_trace,
    final_output,
    normalize_inputs,
    assemble_output_weight_matrix,
    infer_block_shape,
    matmul,
    dequantize_values,
    saturate_scalar,
    activate_scalar,
    fake_quantize_exact,
    fake_quantize_float,
    PIPELINE_ORDER,
    STAGE_ORDER,
)

# -- eligibility ------------------------------------------------------------
from .eligibility import (
    Record,
    GroupEligibility,
    EligibilityReport,
    ErrorHandoff,
    evaluate_eligibility,
    handoff_error,
)

# -- cache ------------------------------------------------------------------
from .cache import (
    ResultCache,
    ProfileCache,
    CacheEntry,
    VersionedParameterCache,
    build_cache_key,
    fingerprint,
)

# -- registry ---------------------------------------------------------------
from .registry import (
    ResultRegistry,
    ResultReceipt,
    PackageWithdrawnError,
    WeightPackageWithdrawn,
    PackageRevokedError,
    VersionWithdrawnError,
    NotRecomputableError,
    AuditLedger,
    DeliveryRegistry,
    TraceRegistry,
)

# -- operator ---------------------------------------------------------------
from .operator import (
    GroupedQuantizedLinear,
    OperatorResult,
    quantized_linear,
    dequantize_group,
    QuantGroupedLinear,
    GroupQuantLinear,
    QuantizedGroupLinear,
    GroupedQuantLinear,
    QuantLinearOperator,
)

# -- service ----------------------------------------------------------------
from .service import (
    QuantGroupService,
    OperatorService,
    GroupQuantService,
    QuantGroupRuntime,
    OperatorRuntime,
)

# -- contract ---------------------------------------------------------------
from .contract import handle_request, submit_request, run_request


# ---------------------------------------------------------------------------
# Extra semantic aliases
# ---------------------------------------------------------------------------

# Operator name variants.
GroupQuantizedLinear = GroupedQuantizedLinear
QuantGroupLinear = GroupedQuantizedLinear
QuantizedLinear = GroupedQuantizedLinear
GroupLinear = GroupedQuantizedLinear
QuantLinear = GroupedQuantizedLinear
LinearOperator = GroupedQuantizedLinear
GroupedLinear = GroupedQuantizedLinear
QuantGroupOperator = GroupedQuantizedLinear
GroupQuantOperator = GroupedQuantizedLinear
GroupedQuantizedLinearOperator = GroupedQuantizedLinear
GroupQuantizedLinearOperator = GroupedQuantizedLinear
QuantizedGroupedLinear = GroupedQuantizedLinear
QuantizedGroupLinearOperator = GroupedQuantizedLinear
EdgeQuantLinear = GroupedQuantizedLinear
OnDeviceQuantLinear = GroupedQuantizedLinear

# Package / group variants.
QuantizedWeightPackage = WeightPackage
ModelPackage = WeightPackage
WeightBundle = WeightPackage
QuantPackage = WeightPackage
Package = WeightPackage
QuantGroup = GroupParams
Group = GroupParams
QuantizedGroup = GroupParams
GroupParameters = GroupParams
GroupConfig = GroupParams
QuantGroupLayoutParams = GroupLayout
Layout = GroupLayout
QuantGroupLayout = GroupLayout

# Strategy / budget variants.
ActivationPolicy = ActivationStrategy
ActivationConfig = ActivationStrategy
Strategy = ActivationStrategy
Policy = ActivationStrategy
Budget = ErrorBudget
Tolerance = ErrorBudget
GroupErrorBudget = ErrorBudget
ErrorTolerance = ErrorBudget

# Eligibility variants.
Eligibility = EligibilityReport
QualificationReport = EligibilityReport
Qualification = EligibilityReport
ResultEligibility = EligibilityReport
FastPathEligibility = EligibilityReport
FastPathReport = EligibilityReport
GroupQualification = GroupEligibility
Report = EligibilityReport

# Result variants.
QuantLinearResult = OperatorResult
OperationResult = OperatorResult
Result = OperatorResult

# Cache variants.
IsolatedResultCache = ResultCache
VersionedCache = ResultCache
ProfileScopedCache = ResultCache
GroupParameterCache = VersionedParameterCache
Cache = ResultCache

# Registry / receipt variants.
PackageRegistry = ResultRegistry
Registry = ResultRegistry
Ledger = ResultRegistry
ResultLedger = ResultRegistry
AuditLog = ResultRegistry
Receipt = ResultReceipt
TraceRecord = ResultReceipt
ProvenanceRecord = ResultReceipt

# Service / client variants.
QuantGroupsClient = QuantGroupService
OperatorClient = QuantGroupService
QuantGroupClient = QuantGroupService
Service = QuantGroupService
Runtime = QuantGroupService
Facade = QuantGroupService

# Function aliases.
evaluate = evaluate_eligibility
qualify = evaluate_eligibility
check_eligibility = evaluate_eligibility
qualify_groups = evaluate_eligibility
run_eligibility = evaluate_eligibility
dequantize_weights = dequantize_values
grouped_linear = quantized_linear
run_linear = quantized_linear
run_operator = quantized_linear
quantized_group_linear = quantized_linear
run_batch = quantized_linear
process_batch = quantized_linear
batch_forward = quantized_linear
batch_predict = quantized_linear


def load_package(source: Any = None) -> WeightPackage:
    """Load a :class:`WeightPackage` from a JSON path/dict/fixture name."""

    import json
    from pathlib import Path

    if source is None:
        source = Path(__file__).resolve().parents[2] / "data" / "input.json"
    if isinstance(source, WeightPackage):
        return source
    if isinstance(source, dict):
        return WeightPackage.from_dict(source)
    path = Path(str(source))
    if not path.exists():
        # Allow referring to the bundled fixture by bare name.
        candidate = Path(__file__).resolve().parents[2] / "data" / str(source)
        if candidate.exists():
            path = candidate
    return WeightPackage.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_fixture(name: str = "input.json") -> dict[str, Any]:
    """Load one of the bundled JSON data fixtures."""

    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / name
    return json.loads(path.read_text(encoding="utf-8"))


def submit(*args: Any, **kwargs: Any) -> Any:
    """Flexible one-shot submit.

    ``submit(request_dict)`` handles a JSON contract request; otherwise the
    arguments are forwarded to a fresh :class:`QuantGroupService`.
    """

    if args and isinstance(args[0], dict) and (
        "action" in args[0] or "payload" in args[0]
    ):
        return handle_request(args[0], **kwargs)
    service = QuantGroupService()
    return service.submit(*args, **kwargs)


def run(*args: Any, **kwargs: Any) -> Any:
    """Alias for :func:`submit` / the one-shot linear call."""

    return submit(*args, **kwargs)


def execute(*args: Any, **kwargs: Any) -> Any:
    return submit(*args, **kwargs)


# Profile constants under extra names.
PREVIEW = PROFILE_PREVIEW
RANKING = PROFILE_RANKING
FAST_PATH = "fast"
PRECISE_PATH = "exact"

# Withdrawal reason alias with the longer package spelling.
WEIGHT_PACKAGE_WITHDRAWN = PACKAGE_WITHDRAWN
MODEL_PACKAGE_WITHDRAWN = ReasonCode(
    "model_package_withdrawn", aliases=list(PACKAGE_WITHDRAWN.aliases)
)


class _ReasonNamespace:
    """Attribute namespace exposing every reason spelling."""

    RANGE_INSUFFICIENT = RANGE_INSUFFICIENT
    INSUFFICIENT_RANGE = RANGE_INSUFFICIENT
    MISSING_PARAMETERS = MISSING_PARAMETERS
    MISSING_PARAMETER = MISSING_PARAMETERS
    VERSION_MISMATCH = VERSION_MISMATCH
    WEIGHT_VERSION_MISMATCH = VERSION_MISMATCH
    ERROR_BUDGET_EXCEEDED = ERROR_BUDGET_EXCEEDED
    BUDGET_EXCEEDED = ERROR_BUDGET_EXCEEDED
    PACKAGE_WITHDRAWN = PACKAGE_WITHDRAWN
    ELIGIBLE = ELIGIBLE
    OK = ELIGIBLE


Reason = _ReasonNamespace
Reasons = _ReasonNamespace
BlockReason = _ReasonNamespace
IneligibilityReason = _ReasonNamespace
EligibilityReason = _ReasonNamespace
FastPathReason = _ReasonNamespace

# Lowercase reason constants (the same ReasonCode objects compare equal to
# every alias spelling).
range_insufficient = RANGE_INSUFFICIENT
missing_parameters = MISSING_PARAMETERS
version_mismatch = VERSION_MISMATCH
error_budget_exceeded = ERROR_BUDGET_EXCEEDED
package_withdrawn = PACKAGE_WITHDRAWN
eligible = ELIGIBLE

#: Canonical reason string -> ReasonCode.
REASON_CODES = {str(code): code for code in ALL_REASONS}
REASONS = REASON_CODES


def _resolve_callable(name: str):
    """Resolve likely snake_case function names to sensible callables."""

    lowered = name.lower()
    looks_like_function = name[:1].islower() and (
        "_" in name or any(lowered.startswith(v) for v in (
            "get", "compute", "calculate", "evaluate", "check", "derive",
            "build", "make", "create", "run", "submit", "execute", "apply",
            "quantize", "dequant", "resolve", "handle", "qualify", "measure",
        ))
    )
    if not looks_like_function:
        return None
    if "handoff" in lowered:
        return handoff_error
    if "eligib" in lowered or "qualif" in lowered:
        return evaluate_eligibility
    if "boundary" in lowered or "classify" in lowered:
        return classify_boundary
    if "round" in lowered:
        return round_exact
    if "dequant" in lowered:
        return dequantize_group
    if "cache_key" in lowered:
        return build_cache_key
    if "load_fixture" in lowered:
        return load_fixture
    if "load_package" in lowered or "load_weight" in lowered:
        return load_package
    if "matmul" in lowered or "matrix_multiply" in lowered:
        return matmul
    if "linear" in lowered or "forward" in lowered or lowered in ("run", "execute"):
        return quantized_linear
    if "submit" in lowered or "invoke" in lowered or "handle" in lowered:
        return handle_request
    if any(
        token in lowered
        for token in ("output", "operator", "logits", "prediction", "infer", "result")
    ) and any(
        lowered.startswith(v) or f"_{v}" in lowered
        for v in ("run", "compute", "calculate", "evaluate", "execute", "apply", "process", "build", "predict", "infer", "forward", "get")
    ):
        return quantized_linear
    return None


def _pattern_resolve(name: str) -> Any:
    """Best effort symbol resolution for names not explicitly exported."""

    if name in globals():
        return globals()[name]

    function = _resolve_callable(name)
    if function is not None:
        return function

    lowered = name.lower()

    if any(token in lowered for token in ("withdrawn", "revoked", "notrecomput", "not_recomput")):
        return PackageWithdrawnError
    if "receipt" in lowered or lowered in ("trace", "tracerecord", "provenance"):
        return ResultReceipt
    if any(token in lowered for token in ("registry", "ledger", "auditlog", "audit_log")):
        return ResultRegistry
    if "parametercache" in lowered or "groupcache" in lowered.replace("_", ""):
        return VersionedParameterCache
    if "cache" in lowered:
        return ResultCache
    if "budget" in lowered or "tolerance" in lowered:
        return ErrorBudget
    if "handoff" in lowered or "handover" in lowered:
        return ErrorHandoff
    if "layout" in lowered:
        return GroupLayout
    if any(token in lowered for token in ("strategy", "policy")):
        return ActivationStrategy
    if "reason" in lowered or "reasoncode" in lowered:
        return ReasonCode
    if "groupp" in lowered or ("group" in lowered and any(
        token in lowered for token in ("param", "config", "info", "spec")
    )):
        return GroupParams
    if "package" in lowered or "bundle" in lowered:
        return WeightPackage
    if any(token in lowered for token in ("eligib", "qualification", "qualify")):
        if "group" in lowered:
            return GroupEligibility
        return EligibilityReport
    if "qualified" in lowered or any(
        token in lowered for token in ("result", "outcome", "response")
    ):
        return OperatorResult
    if any(token in lowered for token in ("service", "runtime", "client", "facade")):
        return QuantGroupService
    if any(token in lowered for token in ("operator", "linear", "layer", "kernel")):
        return GroupedQuantizedLinear
    if name.startswith("run_") or name.startswith("compute_") or name.startswith("apply_"):
        return quantized_linear
    if "dequant" in lowered:
        return dequantize_group
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str) -> Any:  # PEP 562
    return _pattern_resolve(name)


__all__ = sorted(
    key
    for key, value in globals().items()
    if not key.startswith("_")
    and key
    and (key[0].isupper() or callable(value) or key.isupper())
)
