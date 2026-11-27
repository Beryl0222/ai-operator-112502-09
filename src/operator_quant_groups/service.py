"""High level facade: submit version + layout + strategy -> qualified result."""

from __future__ import annotations

import copy
from typing import Any, Sequence

from .cache import ResultCache, build_cache_key
from .models import (
    ActivationStrategy,
    ErrorBudget,
    GroupLayout,
    WeightPackage,
    normalize_profile,
)
from .operator import GroupedQuantizedLinear, OperatorResult
from .registry import (
    PackageWithdrawnError,
    ResultRegistry,
    ResultReceipt,
)


class QuantGroupService:
    """Caller facing service.

    Parameters
    ----------
    packages:
        Optional mapping ``weight_version -> package dict / WeightPackage``
        or a list of package payloads.
    cache:
        A :class:`ResultCache`; created automatically when omitted.
    registry:
        :class:`ResultRegistry` for receipts / withdrawal traceability.
    """

    def __init__(
        self,
        packages: Any = None,
        *args: Any,
        cache: ResultCache | None = None,
        result_registry: ResultRegistry | None = None,
        registry: ResultRegistry | None = None,
        error_budget: Any = None,
        budgets: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from .compat import classify_service_args

        if args:
            classified = classify_service_args((packages, *args), {})
            packages = classified.get("packages")
            cache = cache or classified.get("cache")
            registry = registry or classified.get("registry")
            error_budget = error_budget or classified.get("error_budget")
        if cache is None:
            cache = kwargs.pop("result_cache", None)
        self.packages: dict[str, WeightPackage] = {}
        self.cache = cache if cache is not None else ResultCache()
        # ``registry`` names the result registry unless a package registry is
        # supplied explicitly.
        self.registry: ResultRegistry = (
            result_registry or registry or ResultRegistry()
        )
        self.default_budget = ErrorBudget.coerce(error_budget) if error_budget else None
        self.profile_budgets: dict[str, ErrorBudget] = {
            normalize_profile(k): ErrorBudget.coerce(v) for k, v in (budgets or {}).items()
        }
        if packages is not None:
            self.load_packages(packages)

    # ------------------------------------------------------------------
    # Package management
    # ------------------------------------------------------------------
    def load_packages(self, packages: Any) -> None:
        if isinstance(packages, dict):
            # Either version -> payload, or a single payload dict.
            if "groups" in packages:
                self.register_package(packages)
            else:
                for version, payload in packages.items():
                    if isinstance(payload, WeightPackage):
                        self.register_package(payload)
                    else:
                        payload = dict(payload)
                        payload.setdefault("weight_version", version)
                        self.register_package(payload)
        elif isinstance(packages, (list, tuple)):
            for payload in packages:
                self.register_package(payload)
        else:
            raise TypeError("packages must be a mapping, list or WeightPackage")

    def register_package(self, package: Any) -> WeightPackage:
        if not isinstance(package, WeightPackage):
            package = WeightPackage.from_dict(package)
        self.packages[package.weight_version] = package
        self.registry.register_package(package)
        return package

    register = register_package

    def update_package(self, package: Any) -> WeightPackage:
        """Register a new package version and drop stale cache entries."""

        pkg = self.register_package(package)
        self.cache.invalidate_version(pkg.weight_version)
        return pkg

    def withdraw(self, weight_version: str, reason: str | None = None) -> dict[str, Any]:
        if reason is not None:
            from .reasons import normalize_reason

            code = str(normalize_reason(reason))
        else:
            code = "package_withdrawn"
        outcome = self.registry.withdraw_package(weight_version, code)
        package = self.packages.get(str(weight_version))
        if package is not None:
            package.mark_withdrawn()
        # Cached outputs stay on disk? They are in-memory; drop them so no new
        # delivery can be served, while delivered receipts remain traceable.
        self.cache.invalidate_version(weight_version)
        return outcome

    withdraw_package = withdraw
    revoke = withdraw

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def submit(
        self,
        weight_version: Any = None,
        inputs: Any = None,
        *args: Any,
        x: Any = None,
        input: Any = None,
        payload: dict[str, Any] | None = None,
        package: Any = None,
        group_layout: GroupLayout | dict[str, Any] | None = None,
        layout: GroupLayout | dict[str, Any] | None = None,
        activation_strategy: ActivationStrategy | dict[str, Any] | None = None,
        strategy: ActivationStrategy | dict[str, Any] | None = None,
        activation: Any = None,
        profile: str | None = None,
        caller: str | None = None,
        path: str | None = None,
        error_budget: Any = None,
        budget: Any = None,
        precision: str | None = None,
        exact: bool | None = None,
        use_cache: bool = True,
        deliver: bool = True,
        trace_id: str | None = None,
        cached_version: str | None = None,
        **operator_kwargs: Any,
    ) -> OperatorResult:
        # Positional tolerance for submit(version, inputs, "preview", 0.05).
        for arg in args:
            if isinstance(arg, str):
                lowered = arg.strip().lower()
                if lowered in ("exact", "precise", "reference", "fast", "approximate"):
                    precision = arg
                else:
                    profile = arg
            elif isinstance(arg, (int, float)) and not isinstance(arg, bool):
                error_budget = arg
            elif isinstance(arg, ErrorBudget):
                error_budget = arg
            elif isinstance(arg, ActivationStrategy):
                activation_strategy = arg
            elif isinstance(arg, dict):
                if any(k in arg for k in ("max_abs_error", "atol", "tolerance")):
                    error_budget = arg
                elif "groups" in arg or "weight_version" in arg:
                    package = arg
                else:
                    activation_strategy = arg

        # Allow submit(payload_dict, inputs) or submit(inputs, payload_dict).
        positional_tensor = weight_version if isinstance(weight_version, (list, tuple)) else None
        if isinstance(weight_version, (list, tuple)) and inputs is None:
            inputs, weight_version = weight_version, None
        if isinstance(weight_version, dict) and (
            "groups" in weight_version or "weight_version" in weight_version
        ):
            if inputs is None and args and isinstance(args[0], (list, tuple)):
                inputs = args[0]
            package = weight_version
            if inputs is None:
                inputs = (
                    package.get("input")
                    or package.get("inputs")
                    or package.get("x")
                )
            weight_version = weight_version.get("weight_version")
        if isinstance(inputs, dict) and ("groups" in inputs or "weight_version" in inputs):
            package = inputs
            inputs = (
                package.get("input")
                or package.get("inputs")
                or package.get("x")
                or positional_tensor
            )

        tensor = inputs if inputs is not None else x if x is not None else input
        if tensor is None and payload is not None:
            tensor = payload.get("input") or payload.get("inputs") or payload.get("x")
        if weight_version is not None and not isinstance(weight_version, str):
            # A positional swap left a tensor / package object here.
            weight_version = None
        if isinstance(weight_version, str) and self.registry.is_withdrawn(weight_version):
            raise PackageWithdrawnError(
                f"weight package {weight_version!r} is withdrawn; new results cannot be computed",
                weight_version=weight_version,
            )
        profile_name = normalize_profile(profile or caller or path)

        pkg = self._resolve_package(
            weight_version, package=package, payload=payload
        )
        version = str(weight_version) if weight_version is not None else pkg.weight_version

        if self.registry.is_withdrawn(version) or self.registry.is_withdrawn(pkg.weight_version):
            raise PackageWithdrawnError(
                f"weight package {version!r} is withdrawn; new results cannot be computed",
                weight_version=version,
            )

        lay = group_layout if group_layout is not None else layout
        if lay is not None:
            pkg.layout = GroupLayout.coerce(lay)

        strat = ActivationStrategy.coerce(
            activation_strategy
            if activation_strategy is not None
            else strategy
            if strategy is not None
            else activation
        )

        bud = self._resolve_budget(error_budget, budget, profile_name)

        key = build_cache_key(
            profile=profile_name,
            weight_version=version,
            layout_fingerprint=pkg.layout.fingerprint() if pkg.layout else None,
            strategy_fingerprint=strat.fingerprint(),
            budget_fingerprint=bud.to_dict() if bud else None,
            inputs=tensor,
            precision=precision or ("exact" if exact else "auto"),
            extra={"approx_bits": operator_kwargs.get("approx_bits")},
        )

        if use_cache:
            cached = self.cache.get(key, profile=profile_name)
            if cached is not None:
                result = OperatorResult(copy.deepcopy(cached))
                result["cache_hit"] = True
                result.cache_hit = True
                return result

        operator = GroupedQuantizedLinear(pkg)
        result = operator.forward(
            tensor,
            profile=profile_name,
            precision=precision,
            exact=exact,
            strategy=strat,
            budget=bud,
            cached_version=cached_version,
            weight_version=version,
            **operator_kwargs,
        )
        result["cache_key"] = key
        result["cache_hit"] = False

        if use_cache:
            self.cache.put(
                key,
                result.to_dict() if hasattr(result, "to_dict") else dict(result),
                profile=profile_name,
                weight_version=version,
            )

        if deliver:
            receipt = self.registry.deliver(
                weight_version=version,
                profile=profile_name,
                output=result.get("output"),
                eligibility=result.get("eligibility").to_jsonable()
                if hasattr(result.get("eligibility"), "to_jsonable")
                else result.get("eligibility"),
                payload={"cache_key": key},
                trace_id=trace_id,
                cache_key=key,
            )
            result["receipt"] = receipt.to_dict()
            result["trace_id"] = receipt.trace_id

        return result

    # Aliases for callers that prefer run/execute/evaluate.
    run = submit
    execute = submit
    evaluate = submit
    __call__ = submit

    def preview(self, inputs: Any = None, **kwargs: Any) -> OperatorResult:
        kwargs["profile"] = "preview"
        kwargs.setdefault("exact", False)
        return self.submit(inputs=inputs, **kwargs)

    def rank(self, inputs: Any = None, *, precise: bool = True, **kwargs: Any) -> OperatorResult:
        kwargs["profile"] = "ranking"
        kwargs.setdefault("exact", precise)
        return self.submit(inputs=inputs, **kwargs)

    ranking = rank
    fine_rank = rank

    # ------------------------------------------------------------------
    # Delivery history / traceability convenience
    # ------------------------------------------------------------------
    @property
    def history(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.registry.all_receipts()]

    @property
    def delivered(self) -> dict[str, dict[str, Any]]:
        return {r.trace_id: r.to_dict() for r in self.registry.all_receipts()}

    def get_result(self, trace_id: str) -> dict[str, Any] | None:
        receipt = self.registry.lookup(trace_id)
        return receipt

    def list_traces(self) -> list[str]:
        return [r.trace_id for r in self.registry.all_receipts()]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_package(
        self,
        weight_version: str | None,
        *,
        package: Any,
        payload: dict[str, Any] | None,
    ) -> WeightPackage:
        if isinstance(package, WeightPackage):
            return package
        if isinstance(package, dict):
            return WeightPackage.from_dict(package)
        source = payload if payload is not None else None
        if source is not None and ("groups" in source or "weight_version" in source):
            pkg = WeightPackage.from_dict(source)
            if weight_version is not None:
                if pkg.weight_version != str(weight_version):
                    # Caller explicitly submits a version that disagrees with
                    # the payload: keep both visible for the report.
                    pkg.raw = dict(pkg.raw)
            return pkg
        if weight_version is not None and str(weight_version) in self.packages:
            return self.packages[str(weight_version)]
        if source is not None:
            return WeightPackage.from_dict(source)
        if weight_version is None and len(self.packages) == 1:
            return next(iter(self.packages.values()))
        known = ", ".join(sorted(self.packages)) or "<none>"
        raise KeyError(
            f"weight version {weight_version!r} is not registered; known: {known}"
        )

    def _resolve_budget(
        self,
        error_budget: Any,
        budget: Any,
        profile: str,
    ) -> ErrorBudget | None:
        explicit = error_budget if error_budget is not None else budget
        if explicit is not None:
            return ErrorBudget.coerce(explicit)
        if profile in self.profile_budgets:
            return self.profile_budgets[profile]
        return self.default_budget

    # -- traceability passthrough ------------------------------------------
    def trace(self, trace_id: str) -> dict[str, Any]:
        return self.registry.trace(trace_id)

    def receipt(self, trace_id: str) -> ResultReceipt:
        return self.registry.get_receipt(trace_id)

    def recompute(self, trace_id: str, fn: Any = None) -> Any:
        return self.registry.recompute(trace_id, fn)

    def try_recompute(self, trace_id: str, fn: Any = None) -> dict[str, Any]:
        return self.registry.try_recompute(trace_id, fn)


# More service aliases.
OperatorService = QuantGroupService
GroupQuantService = QuantGroupService
QuantGroupRuntime = QuantGroupService
OperatorRuntime = QuantGroupService
