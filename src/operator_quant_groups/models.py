"""Domain models: groups, layouts, weight packages and strategies."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .numbers import Number, as_fraction

# ---------------------------------------------------------------------------
# Group layout
# ---------------------------------------------------------------------------


@dataclass
class GroupLayout:
    """Describes how quantized group payloads map onto a weight matrix.

    Parameters
    ----------
    group_size:
        Number of output channels covered by one group (default layout).
    in_features / out_features:
        Optional explicit matrix dimensions.  ``out_features`` defaults to
        ``len(groups) * group_size`` and ``in_features`` is inferred from the
        runtime input width when omitted.
    group_axis:
        ``"output"`` (default) means groups partition output channels;
        ``"input"`` means groups partition input channels.
    order:
        Flat value ordering inside a group block; only ``"row_major"`` is
        currently defined.
    """

    group_size: int | None = None
    in_features: int | None = None
    out_features: int | None = None
    group_axis: str = "output"
    order: str = "row_major"
    layout_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.group_size is not None:
            self.group_size = int(self.group_size)
        if self.in_features is not None:
            self.in_features = int(self.in_features)
        if self.out_features is not None:
            self.out_features = int(self.out_features)
        self.group_axis = str(self.group_axis or "output").strip().lower()
        if self.group_axis in ("out", "outputs", "output_channels", "oc"):
            self.group_axis = "output"
        elif self.group_axis in ("in", "inputs", "input_channels", "ic"):
            self.group_axis = "input"
        self.order = str(self.order or "row_major").strip().lower()

    # -- serialisation ------------------------------------------------------
    def fingerprint(self) -> tuple[Any, ...]:
        return (
            "GroupLayout",
            self.group_size,
            self.in_features,
            self.out_features,
            self.group_axis,
            self.order,
            self.layout_version,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "group_size": self.group_size,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "group_axis": self.group_axis,
            "order": self.order,
        }
        if self.layout_version is not None:
            data["layout_version"] = self.layout_version
        data.update(self.extra)
        return data

    @classmethod
    def coerce(cls, value: Any, **overrides: Any) -> "GroupLayout":
        if isinstance(value, GroupLayout):
            if overrides:
                data = value.to_dict()
                data.update(overrides)
                return cls.from_dict(data)
            return value
        if value is None:
            return cls(**overrides)
        if isinstance(value, int):
            return cls(group_size=value, **overrides)
        if isinstance(value, dict):
            data = dict(value)
            data.update(overrides)
            return cls.from_dict(data)
        raise TypeError(f"unsupported group layout: {value!r}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GroupLayout":
        known = {
            "group_size",
            "in_features",
            "out_features",
            "group_axis",
            "order",
            "layout_version",
        }
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in data.items():
            if key in known:
                kwargs[key] = value
            elif key in ("size", "channels_per_group", "channels") and "group_size" not in kwargs:
                kwargs["group_size"] = value
            elif key in ("input_size", "in_dim", "k") and "in_features" not in kwargs:
                kwargs["in_features"] = value
            elif key in ("output_size", "out_dim", "n") and "out_features" not in kwargs:
                kwargs["out_features"] = value
            else:
                extra[key] = value
        kwargs["extra"] = extra
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Quantized group parameters
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = ("scale", "zero_point")


@dataclass
class GroupParams:
    index: int
    group_id: Any = None
    scale: Number | None = None
    zero_point: int | None = None
    values: list[Any] | None = None
    activation_range: tuple[Number, Number] | None = None
    activation_min: Number | None = None
    activation_max: Number | None = None
    weight_version: str | None = None
    bits: int | None = None
    qmin: int | None = None
    qmax: int | None = None
    name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.group_id is None:
            self.group_id = self.index
        if self.zero_point is not None:
            self.zero_point = int(self.zero_point)
        if self.activation_range is not None:
            lo, hi = self.activation_range
            self.activation_range = (lo, hi)
            if self.activation_min is None:
                self.activation_min = lo
            if self.activation_max is None:
                self.activation_max = hi

    # -- presence helpers ---------------------------------------------------
    def missing_fields(self, require_activation_range: bool = True) -> list[str]:
        missing: list[str] = []
        if self.scale is None:
            missing.append("scale")
        if self.zero_point is None:
            missing.append("zero_point")
        if self.values is None:
            missing.append("values")
        if require_activation_range and self.resolved_activation_range() is None:
            missing.append("activation_range")
        return missing

    def resolved_activation_range(
        self, default: Sequence[Number] | None = None
    ) -> tuple[Number, Number] | None:
        if self.activation_range is not None:
            lo, hi = self.activation_range
            if lo is not None and hi is not None:
                return (lo, hi)
        if self.activation_min is not None and self.activation_max is not None:
            return (self.activation_min, self.activation_max)
        if default is not None and len(default) == 2:
            return (default[0], default[1])
        return None

    # -- (de)serialisation --------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int = 0) -> "GroupParams":
        activation_range = data.get("activation_range") or data.get("act_range")
        if activation_range is not None:
            activation_range = tuple(activation_range)  # type: ignore[assignment]
        group_id = data.get("group_id", data.get("id", data.get("name")))
        known = {
            "scale",
            "zero_point",
            "values",
            "activation_range",
            "activation_min",
            "activation_max",
            "weight_version",
            "bits",
            "qmin",
            "qmax",
            "name",
            "group_id",
            "id",
            "act_range",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            index=index,
            group_id=group_id if group_id is not None else index,
            scale=data.get("scale"),
            zero_point=data.get("zero_point", data.get("zp")),
            values=list(data["values"]) if data.get("values") is not None else None,
            activation_range=activation_range,  # type: ignore[arg-type]
            activation_min=data.get("activation_min", data.get("min")),
            activation_max=data.get("activation_max", data.get("max")),
            weight_version=data.get("weight_version", data.get("version")),
            bits=data.get("bits", data.get("num_bits")),
            qmin=data.get("qmin"),
            qmax=data.get("qmax"),
            name=data.get("name"),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "group_id": self.group_id,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "values": list(self.values) if self.values is not None else None,
            "weight_version": self.weight_version,
        }
        if self.activation_range is not None:
            data["activation_range"] = list(self.activation_range)
        if self.bits is not None:
            data["bits"] = self.bits
        data.update(self.extra)
        return data

    # -- numeric ------------------------------------------------------------
    def dequantized_values(self) -> list[Any]:
        """Return ``scale * (q - zero_point)`` values (exact fractions)."""

        if self.scale is None or self.zero_point is None or self.values is None:
            raise ValueError(f"group {self.group_id!r} is missing parameters")
        scale = as_fraction(self.scale)
        zp = self.zero_point
        return [scale * (as_fraction(q) - zp) for q in self.values]


# ---------------------------------------------------------------------------
# Weight package
# ---------------------------------------------------------------------------


@dataclass
class WeightPackage:
    weight_version: str
    groups: list[GroupParams] = field(default_factory=list)
    group_size: int | None = None
    activation_range: tuple[Number, Number] | None = None
    layout: GroupLayout | None = None
    bits: int | None = None
    name: str | None = None
    status: str = "active"
    source: str | None = None
    bias: list[Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weight_version = str(self.weight_version)
        if self.group_size is not None:
            self.group_size = int(self.group_size)
        if self.activation_range is not None:
            self.activation_range = tuple(self.activation_range)  # type: ignore[assignment]
        if self.layout is None:
            self.layout = GroupLayout(group_size=self.group_size)
        elif self.group_size is not None and self.layout.group_size is None:
            self.layout.group_size = self.group_size
        if self.status is None:
            self.status = "active"

    # -- loading ------------------------------------------------------------
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WeightPackage":
        version = (
            payload.get("weight_version")
            or payload.get("version")
            or payload.get("package_version")
        )
        if version is None:
            raise ValueError("weight package payload is missing weight_version")
        raw_groups = payload.get("groups", []) or []
        groups = [
            GroupParams.from_dict(group, index=i)
            for i, group in enumerate(raw_groups)
        ]
        activation_range = payload.get("activation_range") or payload.get("act_range")
        if activation_range is not None:
            activation_range = tuple(activation_range)
        layout_payload = payload.get("group_layout") or payload.get("layout")
        if isinstance(layout_payload, GroupLayout):
            layout = layout_payload
        elif isinstance(layout_payload, dict):
            layout = GroupLayout.from_dict(layout_payload)
        else:
            layout = GroupLayout(
                group_size=payload.get("group_size"),
                in_features=payload.get("in_features"),
                out_features=payload.get("out_features"),
                group_axis=payload.get("group_axis", "output"),
            )
        return cls(
            weight_version=str(version),
            groups=groups,
            group_size=payload.get("group_size", layout.group_size),
            activation_range=activation_range,  # type: ignore[arg-type]
            layout=layout,
            bits=payload.get("bits", payload.get("num_bits")),
            name=payload.get("name"),
            source=payload.get("source"),
            bias=payload.get("bias"),
            raw=payload,
        )

    @classmethod
    def load(cls, path: str | Path) -> "WeightPackage":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    # -- queries ------------------------------------------------------------
    @property
    def version(self) -> str:
        return self.weight_version

    def group_ids(self) -> list[Any]:
        return [g.group_id for g in self.groups]

    def get_group(self, group_id: Any) -> GroupParams | None:
        for group in self.groups:
            if group.group_id == group_id or group.index == group_id:
                return group
        return None

    def resolved_range_for(self, group: GroupParams) -> tuple[Number, Number] | None:
        return group.resolved_activation_range(self.activation_range)

    def mark_withdrawn(self) -> None:
        self.status = "withdrawn"

    @property
    def withdrawn(self) -> bool:
        return self.status in ("withdrawn", "revoked", "retired")


# ---------------------------------------------------------------------------
# Error budget
# ---------------------------------------------------------------------------


@dataclass(init=False)
class ErrorBudget:
    """Per-group error budget for a caller path."""

    max_abs_error: Any = None
    mean_abs_error: Any = None
    max_relative_error: Any = None
    strict_range: bool = True

    def __init__(
        self,
        max_abs_error: Any = None,
        *,
        atol: Any = None,
        abs_tol: Any = None,
        tolerance: Any = None,
        error_tolerance: Any = None,
        mean_abs_error: Any = None,
        max_mae: Any = None,
        max_relative_error: Any = None,
        rtol: Any = None,
        strict_range: bool = True,
        **extra: Any,
    ) -> None:
        chosen_abs = (
            max_abs_error
            if max_abs_error is not None
            else atol
            if atol is not None
            else abs_tol
            if abs_tol is not None
            else tolerance
            if tolerance is not None
            else error_tolerance
        )
        self.max_abs_error = chosen_abs
        self.atol = chosen_abs
        self.abs_tol = chosen_abs
        self.tolerance = chosen_abs
        self.mean_abs_error = mean_abs_error if mean_abs_error is not None else max_mae
        self.max_mae = self.mean_abs_error
        self.max_relative_error = (
            max_relative_error if max_relative_error is not None else rtol
        )
        self.rtol = self.max_relative_error
        self.strict_range = strict_range
        self.extra = extra

    @classmethod
    def coerce(cls, value: Any) -> "ErrorBudget":
        if isinstance(value, ErrorBudget):
            return value
        if value is None:
            return cls()
        if isinstance(value, (int, float)):
            return cls(max_abs_error=value)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"unsupported error budget: {value!r}")

    def allows(self, max_abs: Any, mean_abs: Any = 0) -> bool:
        if self.max_abs_error is not None:
            if as_fraction(max_abs) > as_fraction(self.max_abs_error):
                return False
        if self.mean_abs_error is not None:
            if as_fraction(mean_abs) > as_fraction(self.mean_abs_error):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs_error": self.max_abs_error,
            "atol": self.atol,
            "mean_abs_error": self.mean_abs_error,
            "max_relative_error": self.max_relative_error,
            "strict_range": self.strict_range,
        }


# ---------------------------------------------------------------------------
# Activation strategy
# ---------------------------------------------------------------------------


@dataclass(init=False)
class ActivationStrategy:
    """Ordered activation / saturation / rounding policy.

    The fixed stage order is:

    ``linear -> pre_activation_saturation -> activation
              -> post_activation_saturation -> output_rounding``

    Bounds are inclusive and decided with exact arithmetic, so a value exactly
    on a boundary keeps the same qualification under any dtype or chunking.
    """

    name: str = "hardtanh"
    low: Number | None = None
    high: Number | None = None
    activation_range: tuple[Number, Number] | None = None
    pre_activation_range: tuple[Number, Number] | None = None
    bits: int | None = None
    rounding: str = "half_even"
    symmetric: bool = False
    dtype: str | None = "float64"
    chunk_size: int | None = None
    clamp_before_round: bool = True

    def __init__(
        self,
        name: Any = None,
        low: Number | None = None,
        high: Number | None = None,
        activation_range: Sequence[Number] | None = None,
        *,
        min_value: Number | None = None,
        max_value: Number | None = None,
        min: Number | None = None,
        max: Number | None = None,
        range: Sequence[Number] | None = None,
        bounds: Sequence[Number] | None = None,
        pre_activation_range: Sequence[Number] | None = None,
        pre_range: Sequence[Number] | None = None,
        bits: int | None = None,
        num_bits: int | None = None,
        rounding: str = "half_even",
        rounding_mode: str | None = None,
        symmetric: bool = False,
        dtype: str | None = "float64",
        chunk_size: int | None = None,
        clamp_before_round: bool = True,
        **extra: Any,
    ) -> None:
        # Positional convenience:
        #   ActivationStrategy(-1, 1)        -> hardtanh range [-1, 1]
        #   ActivationStrategy(1.0)          -> symmetric range [-1, 1]
        #   ActivationStrategy("relu")       -> named activation
        if isinstance(name, bool):
            pass
        elif isinstance(name, (int, float)):
            first = name
            second = low if isinstance(low, (int, float)) and not isinstance(low, bool) else None
            if high is None and second is not None:
                low, high = first, second
            elif high is None:
                low, high = -abs(first), abs(first)
            else:
                low = first
            name = "hardtanh"
        elif isinstance(name, (list, tuple)) and activation_range is None:
            activation_range = name  # type: ignore[assignment]
            name = "hardtanh"
        self.name = str(name or "hardtanh").lower()
        if self.name in ("clamp", "clip", "hard_tanh", "hard-tanh"):
            self.name = "hardtanh"
        if self.name in ("identity", "none", "linear", ""):
            self.name = "identity"

        chosen_range = activation_range or range or bounds
        if chosen_range is not None:
            self.low, self.high = chosen_range[0], chosen_range[1]
        else:
            self.low = low if low is not None else min_value if min_value is not None else min
            self.high = high if high is not None else max_value if max_value is not None else max
        if self.low is not None and self.high is not None and self.activation_range is None:
            self.activation_range = (self.low, self.high)
        elif self.low is not None and self.high is not None:
            self.activation_range = (self.low, self.high)
        else:
            self.activation_range = None

        pre = pre_activation_range or pre_range
        self.pre_activation_range = tuple(pre) if pre is not None else None  # type: ignore[assignment]
        self.bits = bits if bits is not None else num_bits
        self.rounding = rounding_mode or rounding
        self.rounding_mode = self.rounding
        self.symmetric = symmetric
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.clamp_before_round = clamp_before_round
        self.extra = extra

    # -- helpers ------------------------------------------------------------
    def resolved_range(
        self, default: Sequence[Number] | None = None
    ) -> tuple[Number, Number] | None:
        if self.low is not None and self.high is not None:
            return (self.low, self.high)
        if default is not None and len(default) == 2:
            return (default[0], default[1])
        return None

    def with_range(self, low: Number, high: Number) -> "ActivationStrategy":
        clone = ActivationStrategy(
            name=self.name,
            activation_range=(low, high),
            pre_activation_range=self.pre_activation_range,
            bits=self.bits,
            rounding=self.rounding,
            symmetric=self.symmetric,
            dtype=self.dtype,
            chunk_size=self.chunk_size,
            clamp_before_round=self.clamp_before_round,
        )
        return clone

    def fingerprint(self) -> tuple[Any, ...]:
        return (
            "ActivationStrategy",
            self.name,
            str(self.low),
            str(self.high),
            tuple(str(x) for x in self.pre_activation_range)
            if self.pre_activation_range
            else None,
            self.bits,
            self.rounding,
            self.symmetric,
            self.dtype,
            self.chunk_size,
            self.clamp_before_round,
        )

    @classmethod
    def coerce(cls, value: Any, **overrides: Any) -> "ActivationStrategy":
        if isinstance(value, ActivationStrategy):
            if not overrides:
                return value
            data = value.to_dict()
            data.update(overrides)
            return cls(**data)
        if value is None:
            return cls(**overrides)
        if isinstance(value, (list, tuple)):
            return cls(activation_range=value, **overrides)
        if isinstance(value, dict):
            data = dict(value)
            data.update(overrides)
            return cls(**data)
        if isinstance(value, str):
            return cls(name=value, **overrides)
        raise TypeError(f"unsupported activation strategy: {value!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "activation_range": [self.low, self.high]
            if self.low is not None and self.high is not None
            else None,
            "pre_activation_range": list(self.pre_activation_range)
            if self.pre_activation_range
            else None,
            "bits": self.bits,
            "rounding": self.rounding,
            "symmetric": self.symmetric,
            "dtype": self.dtype,
            "chunk_size": self.chunk_size,
        }


# ---------------------------------------------------------------------------
# Caller profiles (preview vs fine ranking)
# ---------------------------------------------------------------------------

PROFILE_PREVIEW = "preview"
PROFILE_RANKING = "ranking"

PROFILE_ALIASES = {
    "preview": PROFILE_PREVIEW,
    "fast": PROFILE_PREVIEW,
    "draft": PROFILE_PREVIEW,
    "lite": PROFILE_PREVIEW,
    "quick": PROFILE_PREVIEW,
    "proxy": PROFILE_PREVIEW,
    "rank": PROFILE_RANKING,
    "ranking": PROFILE_RANKING,
    "fine_rank": PROFILE_RANKING,
    "fine-rank": PROFILE_RANKING,
    "finerank": PROFILE_RANKING,
    "rerank": PROFILE_RANKING,
    "精排": PROFILE_RANKING,
    "预览": PROFILE_PREVIEW,
}


def normalize_profile(profile: Any) -> str:
    if profile is None:
        return PROFILE_RANKING
    text = str(profile).strip().lower()
    return PROFILE_ALIASES.get(text, text)
