"""Caches isolated by caller profile and bound to weight versions.

The preview and ranking paths MUST NOT overwrite each other's cache entries,
and a cache entry produced for one weight version can never be silently mixed
with another version.  Keys are deterministic fingerprints and every entry
records ``profile`` + ``weight_version``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return repr(value)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()[:24]


def build_cache_key(
    *,
    profile: str,
    weight_version: str,
    layout_fingerprint: Any = None,
    strategy_fingerprint: Any = None,
    budget_fingerprint: Any = None,
    inputs: Any = None,
    precision: str | None = None,
    extra: Any = None,
) -> str:
    """Deterministic key; ``profile`` and ``weight_version`` are explicit."""

    material = {
        "profile": str(profile),
        "weight_version": str(weight_version),
        "layout": list(layout_fingerprint)
        if isinstance(layout_fingerprint, tuple)
        else layout_fingerprint,
        "strategy": list(strategy_fingerprint)
        if isinstance(strategy_fingerprint, tuple)
        else strategy_fingerprint,
        "budget": budget_fingerprint,
        "inputs_hash": fingerprint(inputs) if inputs is not None else None,
        "precision": precision,
        "extra": extra,
    }
    return f"{profile}:{weight_version}:{fingerprint(material)}"


@dataclass
class CacheEntry:
    key: str
    profile: str
    weight_version: str
    value: Any
    created_at: float = field(default_factory=time.time)
    hits: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "profile": self.profile,
            "weight_version": self.weight_version,
            "created_at": self.created_at,
            "hits": self.hits,
            "metadata": dict(self.metadata),
        }


class ProfileCache:
    """A namespace belonging to exactly one caller profile."""

    def __init__(self, profile: str, owner: "ResultCache | None" = None) -> None:
        self.profile = profile
        self._owner = owner
        self._store: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return default
            entry.hits += 1
            return copy.deepcopy(entry.value)

    def get_entry(self, key: str) -> CacheEntry | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            entry.hits += 1
            return entry

    def put(
        self,
        key: str,
        value: Any,
        *,
        weight_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        with self._lock:
            entry = CacheEntry(
                key=key,
                profile=self.profile,
                weight_version=str(weight_version or "unknown"),
                value=copy.deepcopy(value),
                metadata=dict(metadata or {}),
            )
            self._store[key] = entry
            return entry

    set = put

    def __setitem__(self, key: str, value: Any) -> None:
        self.put(key, value)

    def __getitem__(self, key: str) -> Any:
        return self._store[key].value

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def keys(self) -> list[str]:
        return list(self._store.keys())

    def entries(self) -> list[CacheEntry]:
        return list(self._store.values())

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def invalidate_version(self, weight_version: str) -> int:
        with self._lock:
            doomed = [k for k, e in self._store.items() if e.weight_version == str(weight_version)]
            for key in doomed:
                del self._store[key]
            return len(doomed)


class ResultCache:
    """Profile scoped cache.

    ``cache.for_profile("preview")`` and ``cache.for_profile("ranking")`` are
    independent namespaces.  The top level :meth:`get`/:meth:`put` require a
    profile (or a fully qualified key) so the two paths can never collide.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, ProfileCache] = {}
        self._lock = threading.RLock()

    def for_profile(self, profile: str) -> ProfileCache:
        name = str(profile)
        with self._lock:
            if name not in self._profiles:
                self._profiles[name] = ProfileCache(name, owner=self)
            return self._profiles[name]

    # Common alias.
    namespace = for_profile
    profile_cache = for_profile

    @property
    def preview(self) -> ProfileCache:
        return self.for_profile("preview")

    @property
    def ranking(self) -> ProfileCache:
        return self.for_profile("ranking")

    def get(
        self,
        key: str,
        default: Any = None,
        *,
        profile: str | None = None,
    ) -> Any:
        namespace, bare_key = self._resolve(key, profile)
        return namespace.get(bare_key, default)

    def put(
        self,
        key: str,
        value: Any,
        *,
        profile: str | None = None,
        weight_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        namespace, bare_key = self._resolve(key, profile)
        return namespace.put(
            bare_key, value, weight_version=weight_version, metadata=metadata
        )

    set = put

    def __setitem__(self, key: str, value: Any) -> None:
        profile = key.split(":", 1)[0] if isinstance(key, str) and ":" in key else "default"
        self.put(key, value, profile=profile)

    def __getitem__(self, key: str) -> Any:
        profile = key.split(":", 1)[0] if isinstance(key, str) and ":" in key else "default"
        value = self.get(key, profile=profile)
        if value is None:
            raise KeyError(key)
        return value

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        profile = key.split(":", 1)[0] if ":" in key else "default"
        entry = self.get_entry(key, profile=profile)
        return entry is not None

    def get_entry(self, key: str, *, profile: str | None = None) -> CacheEntry | None:
        namespace, bare_key = self._resolve(key, profile)
        return namespace.get_entry(bare_key)

    def _resolve(self, key: str, profile: str | None) -> tuple[ProfileCache, str]:
        if profile is None and isinstance(key, str) and ":" in key:
            maybe_profile = key.split(":", 1)[0]
            if maybe_profile in self._profiles or maybe_profile in ("preview", "ranking"):
                profile = maybe_profile
        if profile is None:
            raise ValueError("a profile is required to access the shared cache")
        return self.for_profile(profile), key

    def invalidate_version(self, weight_version: str) -> dict[str, int]:
        return {
            name: cache.invalidate_version(weight_version)
            for name, cache in self._profiles.items()
        }

    def clear(self) -> None:
        for cache in self._profiles.values():
            cache.clear()

    def profiles(self) -> list[str]:
        return list(self._profiles.keys())

    def stats(self) -> dict[str, dict[str, int]]:
        return {name: {"entries": len(cache)} for name, cache in self._profiles.items()}


class VersionedParameterCache:
    """Cache for per-group parameters with explicit version checks.

    A stale ``get`` never returns mismatching parameters silently: callers can
    use :meth:`lookup` which returns a ``(params, mismatch_info)`` pair.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[Any, str], Any] = {}
        self._lock = threading.RLock()

    def put(self, group_id: Any, weight_version: str, params: Any) -> None:
        with self._lock:
            self._store[(group_id, str(weight_version))] = copy.deepcopy(params)

    def get(self, group_id: Any, weight_version: str) -> Any | None:
        with self._lock:
            return copy.deepcopy(self._store.get((group_id, str(weight_version))))

    def lookup(
        self, group_id: Any, expected_version: str
    ) -> tuple[Any | None, dict[str, Any] | None]:
        """Return matching params, or the newest stale entry + mismatch info."""

        expected_version = str(expected_version)
        match = self.get(group_id, expected_version)
        if match is not None:
            return match, None
        stale_versions = [v for (gid, v) in self._store if gid == group_id]
        if not stale_versions:
            return None, {"reason": "missing_parameters", "group_id": group_id}
        newest = sorted(stale_versions)[-1]
        stale = self.get(group_id, newest)
        return None, {
            "reason": "version_mismatch",
            "group_id": group_id,
            "expected_version": expected_version,
            "actual_version": newest,
            "stale_parameters": stale,
        }
