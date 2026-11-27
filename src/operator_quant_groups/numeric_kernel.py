"""Exact numeric primitives used for boundary and eligibility decisions.

The operator never makes a qualification decision from order dependent
floating point accumulation.  Raw numbers are canonicalised to
:class:`fractions.Fraction` through their shortest decimal spelling so that
``0.1`` means the decimal value 1/10 (the same value an author wrote in JSON),
and all boundary comparisons / reductions are then exact.

The approximate (preview) path may use ordinary floats and simulated dtypes,
but eligibility is always derived from the exact representation so that
changing dtype or chunk size can never flip a boundary decision.
"""

from __future__ import annotations

import math
import struct
from decimal import Decimal
from fractions import Fraction
from typing import Any, Iterable, Iterator, Sequence

Number = int | float | str | Decimal | Fraction

# ---------------------------------------------------------------------------
# Canonical exact numbers
# ---------------------------------------------------------------------------


def as_fraction(value: Any) -> Fraction:
    """Convert *value* to an exact :class:`Fraction`.

    Floats are interpreted through their shortest round-trip decimal
    representation (``repr``) so that author intent such as ``0.1`` is
    preserved.  Strings accept decimal (``"0.25"``), exponent (``"1e-3"``) and
    rational (``"3/8"``) spellings.
    """

    if value is None:
        raise TypeError("cannot convert None to an exact number")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        return Fraction(int(value))
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Decimal):
        return Fraction(value)
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("NaN has no exact boundary representation")
        if math.isinf(value):
            # Infinity is still useful for open ranges.
            return Fraction(1) if value > 0 else Fraction(-1)  # pragma: no cover
        return Fraction(repr(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty numeric string")
        try:
            return Fraction(text)
        except (ValueError, ZeroDivisionError):
            # Fraction does accept exponent strings on modern Python, but be
            # explicit and consistent by going through Decimal first.
            return Fraction(Decimal(text))
    if hasattr(value, "as_integer_ratio"):
        try:
            numerator, denominator = value.as_integer_ratio()
            return Fraction(numerator, denominator)
        except Exception:  # pragma: no cover - defensive
            pass
    raise TypeError(f"unsupported numeric value: {value!r} ({type(value)!r})")


def as_float(value: Any) -> float:
    """Best effort float conversion that keeps booleans honest."""

    if isinstance(value, float):
        return value
    return float(as_fraction(value))


def fraction_to_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


# ---------------------------------------------------------------------------
# Dtype simulation
# ---------------------------------------------------------------------------

#: dtypes that the operator can simulate for approximate outputs.
DTYPE_ALIASES = {
    "float": "float64",
    "double": "float64",
    "fp64": "float64",
    "f64": "float64",
    "float64": "float64",
    "fp32": "float32",
    "f32": "float32",
    "float32": "float32",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}


def normalize_dtype(dtype: str | None) -> str:
    if dtype is None:
        return "float64"
    key = str(dtype).strip().lower()
    return DTYPE_ALIASES.get(key, key)


def cast_scalar(value: Any, dtype: str | None = "float64") -> float:
    """Round a scalar the way *dtype* would.

    Eligibility never depends on this cast; it only shapes approximate output.
    """

    kind = normalize_dtype(dtype)
    x = float(value)
    if kind == "float64":
        return x
    if kind == "float32":
        return struct.unpack("!f", struct.pack("!f", x))[0]
    if kind == "bfloat16":
        # Round to nearest even, keeping the top 16 bits of an fp32 payload.
        packed = struct.unpack("!I", struct.pack("!f", x))[0]
        rounding_bias = 0x7FFF + ((packed >> 16) & 1)
        packed = (packed + rounding_bias) & 0xFFFF0000
        return struct.unpack("!f", struct.pack("!I", packed))[0]
    if kind in ("int", "int32", "int8", "uint8"):
        return float(round(x))
    raise ValueError(f"unsupported dtype simulation: {dtype!r}")


def cast_nested(value: Any, dtype: str | None) -> Any:
    if isinstance(value, (list, tuple)):
        return [cast_nested(item, dtype) for item in value]
    return cast_scalar(value, dtype)


# ---------------------------------------------------------------------------
# Chunking and stable reductions
# ---------------------------------------------------------------------------


def iter_chunks(items: Sequence[Any], chunk_size: int | None) -> Iterator[Sequence[Any]]:
    if chunk_size is None or chunk_size <= 0 or chunk_size >= len(items):
        yield items
        return
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def exact_sum(values: Iterable[Any], chunk_size: int | None = None) -> Fraction:
    """Associative exact sum.

    :class:`Fraction` addition is mathematically associative, so chunking and
    iteration order cannot change the result.
    """

    materialised = list(values)
    total = Fraction(0)
    if chunk_size:
        for chunk in iter_chunks(materialised, chunk_size):
            part = Fraction(0)
            for value in chunk:
                part += as_fraction(value)
            total += part
    else:
        for value in materialised:
            total += as_fraction(value)
    return total


def stable_float_sum(values: Iterable[float], chunk_size: int | None = None) -> float:
    """Chunk tolerant float sum based on :func:`math.fsum`."""

    materialised = [float(v) for v in values]
    if not chunk_size:
        return math.fsum(materialised)
    parts = [
        math.fsum(chunk)
        for chunk in iter_chunks(materialised, chunk_size)  # type: ignore[arg-type]
    ]
    return math.fsum(parts)


def max_abs_error(
    candidate: Sequence[Sequence[Any]] | Sequence[Any],
    reference: Sequence[Sequence[Any]] | Sequence[Any],
    *,
    chunk_size: int | None = None,
) -> Fraction:
    """Maximum absolute element-wise error, computed exactly."""

    worst = Fraction(0)

    def _walk(a: Any, b: Any) -> None:
        nonlocal worst
        if isinstance(a, (list, tuple)):
            for xa, xb in zip(a, b):
                _walk(xa, xb)
        else:
            delta = abs(as_fraction(a) - as_fraction(b))
            if delta > worst:
                worst = delta

    if chunk_size:
        # Chunking only changes traversal; the max reduction is associative.
        flat_a = list(_flatten(candidate))
        flat_b = list(_flatten(reference))
        for ca, cb in zip(
            iter_chunks(flat_a, chunk_size), iter_chunks(flat_b, chunk_size)
        ):
            for a, b in zip(ca, cb):
                delta = abs(as_fraction(a) - as_fraction(b))
                if delta > worst:
                    worst = delta
        return worst

    _walk(candidate, reference)
    return worst


def mean_abs_error(
    candidate: Sequence[Any], reference: Sequence[Any]
) -> Fraction:
    total = Fraction(0)
    count = 0

    def _walk(a: Any, b: Any) -> None:
        nonlocal total, count
        if isinstance(a, (list, tuple)):
            for xa, xb in zip(a, b):
                _walk(xa, xb)
        else:
            total += abs(as_fraction(a) - as_fraction(b))
            count += 1

    _walk(candidate, reference)
    if count == 0:
        return Fraction(0)
    return total / count


def _flatten(value: Any) -> Iterator[Any]:
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


# ---------------------------------------------------------------------------
# Deterministic rounding at quantisation boundaries
# ---------------------------------------------------------------------------

ROUND_MODES = {
    "half_even",
    "nearest_even",
    "bankers",
    "half_up",
    "half_down",
    "half_away",
    "half_away_from_zero",
    "floor",
    "down",
    "ceil",
    "up",
    "trunc",
    "truncate",
}


def round_exact(value: Number, mode: str = "half_even") -> int:
    """Round an exact value to an integer using a named tie rule."""

    x = as_fraction(value)
    mode = (mode or "half_even").strip().lower()
    if mode in ("nearest_even", "bankers"):
        mode = "half_even"
    if mode in ("half_away_from_zero",):
        mode = "half_away"
    if mode in ("down",):
        mode = "floor"
    if mode in ("up",):
        mode = "ceil"
    if mode in ("truncate",):
        mode = "trunc"
    if mode not in ROUND_MODES and mode not in (
        "half_even",
        "half_up",
        "half_down",
        "half_away",
        "floor",
        "ceil",
        "trunc",
    ):
        raise ValueError(f"unknown rounding mode: {mode!r}")

    floor_value = math.floor(x)
    fraction = x - floor_value
    if fraction == 0:
        return floor_value
    if mode == "floor":
        return floor_value
    if mode == "ceil":
        return floor_value + 1
    if mode == "trunc":
        return math.trunc(x)

    half = Fraction(1, 2)
    if fraction < half:
        return floor_value
    if fraction > half:
        return floor_value + 1

    # Exact tie.
    if mode == "half_even":
        return floor_value if floor_value % 2 == 0 else floor_value + 1
    if mode == "half_up":
        return floor_value + 1
    if mode == "half_down":
        return floor_value
    if mode == "half_away":
        if x >= 0:
            return floor_value + 1
        return floor_value  # -1.5 -> -2 (floor), away from zero
    raise ValueError(f"unhandled rounding mode: {mode!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Boundary classification
# ---------------------------------------------------------------------------

BOUNDARY_BELOW = "below_range"
BOUNDARY_AT_LOWER = "at_lower_bound"
BOUNDARY_INSIDE = "inside_range"
BOUNDARY_AT_UPPER = "at_upper_bound"
BOUNDARY_ABOVE = "above_range"


def classify_boundary(
    value: Number,
    low: Number | None,
    high: Number | None,
    *,
    dtype: str | None = None,
    chunk_size: int | None = None,
    epsilon: Number | None = None,
    tolerance: Number | None = None,
) -> str:
    """Classify *value* against ``[low, high]``.

    Bounds are inclusive: a value exactly equal to a bound is reported as the
    corresponding ``at_*_bound`` state, never as outside.

    ``dtype`` / ``chunk_size`` are accepted for API symmetry but do not alter
    the exact decision (that is the determinism guarantee).  An optional
    ``epsilon``/``tolerance`` widens the bounds when a caller explicitly wants
    a forgiving comparison.
    """

    eps = (
        as_fraction(epsilon if epsilon is not None else tolerance)
        if (epsilon is not None or tolerance is not None)
        else Fraction(0)
    )
    v = as_fraction(value)
    lo = None if low is None else as_fraction(low) - eps
    hi = None if high is None else as_fraction(high) + eps
    if lo is not None:
        if v < lo:
            return BOUNDARY_BELOW
        if v == lo:
            return BOUNDARY_AT_LOWER
    if hi is not None:
        if v > hi:
            return BOUNDARY_ABOVE
        if v == hi:
            return BOUNDARY_AT_UPPER
    return BOUNDARY_INSIDE


def within_inclusive(
    value: Number,
    low: Number | None,
    high: Number | None,
    **kwargs: Any,
) -> bool:
    """Inclusive range test using exact arithmetic."""

    state = classify_boundary(value, low, high, **kwargs)
    return state in (
        BOUNDARY_INSIDE,
        BOUNDARY_AT_LOWER,
        BOUNDARY_AT_UPPER,
    )


def overflow_distance(
    value: Number, low: Number | None, high: Number | None
) -> Fraction:
    """How far *value* lies outside ``[low, high]`` (``0`` on the boundary)."""

    v = as_fraction(value)
    distance = Fraction(0)
    if low is not None and v < as_fraction(low):
        distance = max(distance, as_fraction(low) - v)
    if high is not None and v > as_fraction(high):
        distance = max(distance, v - as_fraction(high))
    return distance
