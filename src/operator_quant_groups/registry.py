"""Package registry, delivery receipts and withdrawal traceability.

Delivered results are frozen into receipts.  Withdrawing a weight package marks
the matching receipts as non recomputable but never deletes them: old outputs
stay traceable and every receipt states the precise reason recomputation is
blocked.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .reasons import PACKAGE_WITHDRAWN, ReasonCode, normalize_reason


class PackageWithdrawnError(RuntimeError):
    """Raised when work is attempted against a withdrawn package version."""

    def __init__(
        self,
        message: str,
        *,
        weight_version: str | None = None,
        reason_code: str = str(PACKAGE_WITHDRAWN),
        trace_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.weight_version = weight_version
        self.reason_code = reason_code
        self.trace_id = trace_id
        self.code = reason_code
        self.reason = reason_code


# Common alias names.
WeightPackageWithdrawn = PackageWithdrawnError
PackageRevokedError = PackageWithdrawnError
VersionWithdrawnError = PackageWithdrawnError
NotRecomputableError = PackageWithdrawnError


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ResultReceipt:
    """Immutable-ish trace record for one delivered result."""

    trace_id: str
    weight_version: str
    profile: str
    output: Any = None
    eligibility: dict[str, Any] | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)
    package_status: str = "active"
    recomputable: bool = True
    recompute_blocked: bool = False
    recompute_block_reason: str | None = None
    reason: str | None = None
    withdrawn_at: str | None = None
    cache_key: str | None = None

    def mark_withdrawn(self, reason_code: Any = PACKAGE_WITHDRAWN) -> None:
        # Preserve the ReasonCode object so equality also matches aliases.
        if not isinstance(reason_code, ReasonCode):
            reason_code = normalize_reason(reason_code) or ReasonCode(str(reason_code))
        self.package_status = "withdrawn"
        self.recomputable = False
        self.recompute_blocked = True
        self.recompute_block_reason = reason_code
        # Keep ``reason`` aligned for readers expecting either field.
        self.reason = reason_code
        self.withdrawn_at = _utc_now_iso()

    # -- access helpers -----------------------------------------------------
    @property
    def id(self) -> str:
        return self.trace_id

    @property
    def receipt_id(self) -> str:
        return self.trace_id

    @property
    def unrecomputable_reason(self) -> str | None:
        return self.recompute_block_reason

    @property
    def blocked_reason(self) -> str | None:
        return self.recompute_block_reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "receipt_id": self.trace_id,
            "id": self.trace_id,
            "weight_version": self.weight_version,
            "profile": self.profile,
            "output": self.output,
            "outputs": self.output,
            "eligibility": self.eligibility,
            "payload": self.payload,
            "created_at": self.created_at,
            "package_status": self.package_status,
            "status": self.package_status,
            "recomputable": self.recomputable,
            "can_recompute": self.recomputable,
            "recompute_blocked": self.recompute_blocked,
            "recompute_block_reason": self.recompute_block_reason,
            "unrecomputable_reason": self.recompute_block_reason,
            "reason": self.reason,
            "withdrawn_at": self.withdrawn_at,
            "cache_key": self.cache_key,
        }


class ResultRegistry:
    """Keeps delivered receipts and package withdrawal state."""

    def __init__(self, *, id_factory: Callable[[], str] | None = None) -> None:
        self._receipts: dict[str, ResultReceipt] = {}
        self._by_version: dict[str, list[str]] = {}
        self._packages: dict[str, Any] = {}
        self._withdrawn: set[str] = set()
        self._lock = threading.RLock()
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)

    # -- packages -----------------------------------------------------------
    def register_package(self, package: Any) -> Any:
        version = str(getattr(package, "weight_version", None) or package["weight_version"])
        with self._lock:
            self._packages[version] = package
            if version in self._withdrawn and hasattr(package, "mark_withdrawn"):
                package.mark_withdrawn()
        return package

    def register(self, package: Any) -> Any:
        return self.register_package(package)

    def get_package(self, weight_version: str) -> Any | None:
        return self._packages.get(str(weight_version))

    def is_withdrawn(self, weight_version: str) -> bool:
        return str(weight_version) in self._withdrawn

    def package_status(self, weight_version: str) -> str:
        return "withdrawn" if self.is_withdrawn(weight_version) else "active"

    def withdraw_package(
        self, weight_version: str, reason_code: Any = PACKAGE_WITHDRAWN
    ) -> dict[str, Any]:
        """Withdraw a version and mark all its receipts non recomputable."""

        if not isinstance(reason_code, ReasonCode):
            reason_code = normalize_reason(reason_code) or PACKAGE_WITHDRAWN
        version = str(weight_version)
        with self._lock:
            self._withdrawn.add(version)
            package = self._packages.get(version)
            if package is not None and hasattr(package, "mark_withdrawn"):
                package.mark_withdrawn()
            affected = []
            for trace_id in self._by_version.get(version, []):
                receipt = self._receipts[trace_id]
                receipt.mark_withdrawn(reason_code)
                affected.append(trace_id)
            return {
                "weight_version": version,
                "status": "withdrawn",
                "reason": reason_code,
                "affected_receipts": affected,
                "affected_count": len(affected),
            }

    # Common aliases for withdrawal.
    withdraw = withdraw_package
    revoke = withdraw_package
    revoke_package = withdraw_package
    unregister_package = withdraw_package
    retire = withdraw_package

    # -- receipts -----------------------------------------------------------
    def deliver(
        self,
        *,
        weight_version: str,
        profile: str,
        output: Any = None,
        eligibility: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        trace_id: str | None = None,
        cache_key: str | None = None,
    ) -> ResultReceipt:
        with self._lock:
            tid = trace_id or self._id_factory()
            receipt = ResultReceipt(
                trace_id=tid,
                weight_version=str(weight_version),
                profile=str(profile),
                output=output,
                eligibility=eligibility,
                payload=dict(payload or {}),
                cache_key=cache_key,
            )
            if self.is_withdrawn(receipt.weight_version):
                receipt.mark_withdrawn()
            self._receipts[tid] = receipt
            self._by_version.setdefault(receipt.weight_version, []).append(tid)
            return receipt

    def add_receipt(self, receipt: ResultReceipt) -> ResultReceipt:
        with self._lock:
            self._receipts[receipt.trace_id] = receipt
            self._by_version.setdefault(receipt.weight_version, []).append(receipt.trace_id)
            if self.is_withdrawn(receipt.weight_version):
                receipt.mark_withdrawn()
            return receipt

    def get_receipt(self, trace_id: str) -> ResultReceipt:
        return self._receipts[trace_id]

    def get(self, trace_id: str) -> ResultReceipt:
        return self._receipts[trace_id]

    def trace(self, trace_id: str) -> dict[str, Any]:
        """Trace view: always available, even after withdrawal."""

        return self._receipts[trace_id].to_dict()

    def lookup(self, trace_id: str) -> dict[str, Any] | None:
        receipt = self._receipts.get(trace_id)
        return receipt.to_dict() if receipt else None

    def receipts_for_version(self, weight_version: str) -> list[ResultReceipt]:
        return [self._receipts[t] for t in self._by_version.get(str(weight_version), [])]

    def all_receipts(self) -> list[ResultReceipt]:
        return list(self._receipts.values())

    # -- recomputation ------------------------------------------------------
    def recompute(
        self,
        trace_id: str,
        recompute_fn: Callable[[ResultReceipt], Any] | None = None,
    ) -> Any:
        """Recompute a delivered result unless its package was withdrawn.

        Raises :class:`PackageWithdrawnError` with a stable reason code when
        recomputation is impossible.  If no callback is supplied, the frozen
        delivered output is returned for active packages.
        """

        receipt = self._receipts[trace_id]
        if not receipt.recomputable or self.is_withdrawn(receipt.weight_version):
            if receipt.package_status != "withdrawn":
                receipt.mark_withdrawn()
            raise PackageWithdrawnError(
                f"weight package {receipt.weight_version!r} was withdrawn; "
                "delivered result remains traceable but cannot be recomputed",
                weight_version=receipt.weight_version,
                reason_code=receipt.recompute_block_reason or str(PACKAGE_WITHDRAWN),
                trace_id=trace_id,
            )
        if recompute_fn is not None:
            return recompute_fn(receipt)
        return receipt.output

    def try_recompute(
        self,
        trace_id: str,
        recompute_fn: Callable[[ResultReceipt], Any] | None = None,
    ) -> dict[str, Any]:
        """Non raising recomputation outcome with explicit eligibility."""

        try:
            value = self.recompute(trace_id, recompute_fn)
            return {
                "recomputable": True,
                "ok": True,
                "output": value,
                "trace_id": trace_id,
            }
        except PackageWithdrawnError as exc:
            receipt = self._receipts[trace_id]
            return {
                "recomputable": False,
                "ok": False,
                "output": receipt.output,
                "frozen_output": receipt.output,
                "trace_id": trace_id,
                "weight_version": receipt.weight_version,
                "reason": exc.reason_code,
                "reason_code": exc.reason_code,
                "message": str(exc),
            }


# A light alias for callers that think in terms of an audit ledger.
AuditLedger = ResultRegistry
DeliveryRegistry = ResultRegistry
TraceRegistry = ResultRegistry
