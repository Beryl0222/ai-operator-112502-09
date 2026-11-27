"""Pure compute engine: dequantize -> linear -> saturate/activate -> round.

Two numeric tracks are provided:

* exact track: :class:`fractions.Fraction` everywhere.  Used for eligibility
  decisions and for the ranking precise path;
* float track: ordinary floats with :func:`math.fsum` reductions and optional
  dtype simulation.  Used for the controlled preview approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Sequence

from .models import ActivationStrategy, GroupLayout
from .numbers import (
    as_float,
    as_fraction,
    cast_nested,
    exact_sum,
    round_exact,
    stable_float_sum,
)

# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

#: The single defined stage order.  Boundary semantics depend on this order.
PIPELINE_ORDER = (
    "dequantize",
    "linear",
    "pre_activation_saturation",
    "activation",
    "post_activation_saturation",
    "output_rounding",
)

#: Human readable alias for the stage order.
STAGE_ORDER = PIPELINE_ORDER


def normalize_inputs(inputs: Any) -> tuple[list[list[Any]], bool]:
    """Return ``(matrix, was_vector)``.  Accepts 1-D and 2-D array-likes."""

    if inputs is None:
        raise ValueError("inputs are required")
    if isinstance(inputs, (str, bytes)):
        raise TypeError("inputs must be numeric sequences, not strings")
    if not isinstance(inputs, (list, tuple)):
        # Support generic iterables without depending on numpy.
        try:
            inputs = list(inputs)
        except TypeError as exc:  # pragma: no cover
            raise TypeError("inputs must be a sequence of rows") from exc
    if len(inputs) == 0:
        return [], False
    first = inputs[0]
    if isinstance(first, (list, tuple)):
        return [list(row) for row in inputs], False
    # 1-D single sample.
    return [list(inputs)], True


# ---------------------------------------------------------------------------
# Dequantisation and weight assembly
# ---------------------------------------------------------------------------


def dequantize_values(values: Sequence[Any], scale: Any, zero_point: int) -> list:
    s = as_fraction(scale)
    zp = as_fraction(zero_point)
    return [s * (as_fraction(q) - zp) for q in values]


def infer_block_shape(
    value_count: int,
    group_size: int | None,
    in_features: int | None,
) -> tuple[int, int]:
    """Infer ``(rows, cols)`` of one group block."""

    if group_size and in_features:
        if value_count != group_size * in_features:
            # Be forgiving: derive columns from the actual payload.
            if value_count % group_size == 0:
                return group_size, value_count // group_size
        return group_size, in_features
    if group_size and value_count % group_size == 0:
        return group_size, value_count // group_size
    if in_features and value_count % in_features == 0:
        return value_count // in_features, in_features
    if group_size:
        return group_size, value_count // group_size
    # Last resort: one row per value.
    return value_count, 1


def assemble_output_weight_matrix(
    package: Any,
    in_features: int,
    *,
    exact: bool = True,
) -> tuple[list[list[Any]], list[tuple[int, int]], list[int]]:
    """Build an ``(out_features, in_features)`` W and group row ranges.

    Returns ``(weight_matrix, row_ranges, group_indices)`` where
    ``row_ranges[i] = (start_row, end_row)`` for ``package.groups[i]``.
    """

    layout: GroupLayout | None = getattr(package, "layout", None)
    group_size = getattr(package, "group_size", None)
    if layout is not None and layout.group_size:
        group_size = layout.group_size

    matrix: list[list[Any]] = []
    ranges: list[tuple[int, int]] = []
    indices: list[int] = []

    for i, group in enumerate(package.groups):
        if group.scale is None or group.zero_point is None or group.values is None:
            # Cannot assemble; eligibility report will explain.  Insert a zero
            # block of the best known shape so row ranges stay aligned.
            g = group_size or 1
            block = [[0 for _ in range(in_features)] for _ in range(g)]
        else:
            values = (
                group.dequantized_values()
                if exact
                else [as_float(v) for v in group.dequantized_values()]
            )
            explicit_shape = None
            extra = getattr(group, "extra", None) or {}
            shape_value = extra.get("weight_shape") or extra.get("shape") or extra.get("block_shape")
            if isinstance(shape_value, (list, tuple)) and len(shape_value) == 2:
                explicit_shape = (int(shape_value[0]), int(shape_value[1]))
            if explicit_shape is not None:
                rows, cols = explicit_shape
            else:
                rows, cols = infer_block_shape(
                    len(values),
                    group_size,
                    in_features if in_features else (layout.in_features if layout else None),
                )
            if cols != in_features and in_features and explicit_shape is None:
                # Flatten/reshape defensively.
                cols = in_features
                rows = max(1, len(values) // cols)
            block = [
                values[r * cols : (r + 1) * cols] for r in range(rows) if r * cols < len(values)
            ]
            # Pad a short final row rather than crashing on malformed payloads.
            for row in block:
                if len(row) < in_features:
                    row.extend([0 if exact else 0.0] * (in_features - len(row)))
        start = len(matrix)
        matrix.extend(block)
        end = len(matrix)
        ranges.append((start, end))
        indices.append(i)

    if layout is not None and layout.out_features and len(matrix) < layout.out_features:
        zero = 0 if exact else 0.0
        while len(matrix) < layout.out_features:
            matrix.append([zero] * in_features)
    return matrix, ranges, indices


# ---------------------------------------------------------------------------
# Matrix multiplication
# ---------------------------------------------------------------------------


def matmul(
    inputs: Sequence[Sequence[Any]],
    weights: Sequence[Sequence[Any]],
    *,
    exact: bool = True,
    chunk_size: int | None = None,
) -> list[list[Any]]:
    """``inputs (batch, K) @ weights.T (K, N) -> (batch, N)``."""

    if not inputs:
        return []
    k = len(inputs[0])
    n = len(weights)
    result: list[list[Any]] = []
    for row in inputs:
        out_row: list[Any] = []
        for w_row in weights:
            if exact:
                terms = (
                    as_fraction(row[j]) * as_fraction(w_row[j])
                    for j in range(min(k, len(w_row)))
                )
                value = exact_sum(terms, chunk_size=chunk_size)
            else:
                terms = (row[j] * w_row[j] for j in range(min(k, len(w_row))))
                value = stable_float_sum(
                    (as_float(t) for t in terms), chunk_size=chunk_size
                )
            out_row.append(value)
        result.append(out_row)
    return result


def add_bias(outputs: list[list[Any]], bias: Sequence[Any] | None, exact: bool) -> None:
    if not bias:
        return
    for row in outputs:
        for j in range(min(len(row), len(bias))):
            b = as_fraction(bias[j]) if exact else as_float(bias[j])
            row[j] = row[j] + b if exact else row[j] + b


# ---------------------------------------------------------------------------
# Stage primitives (fixed order)
# ---------------------------------------------------------------------------


def saturate_scalar(value: Any, low: Any, high: Any, *, exact: bool = True) -> Any:
    if exact:
        v = as_fraction(value)
        lo = as_fraction(low) if low is not None else None
        hi = as_fraction(high) if high is not None else None
        if lo is not None and v < lo:
            return lo
        if hi is not None and v > hi:
            return hi
        return v
    v = as_float(value)
    if low is not None and v < as_float(low):
        return as_float(low)
    if high is not None and v > as_float(high):
        return as_float(high)
    return v


def activate_scalar(value: Any, strategy: ActivationStrategy, *, exact: bool = True) -> Any:
    name = strategy.name
    if name in ("identity", "none"):
        return value
    if name == "relu":
        zero = Fraction(0) if exact else 0.0
        return value if value > 0 else zero
    if name == "relu6":
        return saturate_scalar(value, 0, 6, exact=exact)
    if name == "hardtanh":
        rng = strategy.activation_range
        if rng is None:
            return value
        return saturate_scalar(value, rng[0], rng[1], exact=exact)
    if name in ("sigmoid", "tanh", "gelu", "silu"):
        # Transcendental activations are only defined on the float track.
        x = as_float(value)
        if name == "sigmoid":
            return 1.0 / (1.0 + math.exp(-x))
        if name == "tanh":
            return math.tanh(x)
        if name == "silu":
            return x / (1.0 + math.exp(-x))
        if name == "gelu":
            return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
    raise ValueError(f"unsupported activation: {name!r}")


def fake_quantize_exact(
    value: Any,
    low: Any,
    high: Any,
    bits: int,
    *,
    rounding: str = "half_even",
    symmetric: bool = False,
) -> Any:
    """Exact asymmetric fake quantisation over ``[low, high]``."""

    v = as_fraction(value)
    lo = as_fraction(low)
    hi = as_fraction(high)
    levels = 2**bits
    if symmetric:
        # Signed symmetric grid with (levels - 2) positive steps.
        qmax = levels // 2 - 1
        scale = hi / qmax if hi != 0 else Fraction(0)
        if scale == 0:
            return lo
        q = round_exact(v / scale, rounding)
        q = max(-qmax - 1, min(qmax, q))
        return q * scale
    scale = (hi - lo) / (levels - 1)
    if scale == 0:
        return lo
    q = round_exact((v - lo) / scale, rounding)
    q = max(0, min(levels - 1, q))
    return lo + q * scale


def fake_quantize_float(
    value: float,
    low: float,
    high: float,
    bits: int,
    *,
    rounding: str = "half_even",
) -> float:
    levels = 2**bits
    scale = (high - low) / (levels - 1)
    if scale == 0:
        return low
    q = round((value - low) / scale)  # Python round = half even
    q = max(0, min(levels - 1, int(q)))
    return low + q * scale


# ---------------------------------------------------------------------------
# Full traced evaluation
# ---------------------------------------------------------------------------


@dataclass
class ComputeTrace:
    """All stage tensors for one evaluation."""

    pre_activation: list[list[Any]] = field(default_factory=list)
    pre_saturation: list[list[Any]] = field(default_factory=list)
    activated: list[list[Any]] = field(default_factory=list)
    post_saturation: list[list[Any]] = field(default_factory=list)
    rounded: list[list[Any]] = field(default_factory=list)
    output: list[list[Any]] = field(default_factory=list)
    weight_matrix: list[list[Any]] = field(default_factory=list)
    row_ranges: list[tuple[int, int]] = field(default_factory=list)
    in_features: int = 0
    was_vector: bool = False
    exact: bool = True
    dtype: str | None = None

    def group_tensor(self, tensor_name: str, row_range: tuple[int, int]) -> list[list[Any]]:
        tensor = getattr(self, tensor_name)
        start, end = row_range
        return [row[start:end] for row in tensor]


def evaluate_trace(
    inputs: Any,
    package: Any,
    strategy: ActivationStrategy | None = None,
    *,
    exact: bool = True,
    chunk_size: int | None = None,
    dtype: str | None = None,
    approx_bits: int | None = None,
    bias: Sequence[Any] | None = None,
) -> ComputeTrace:
    """Run the fixed ordered pipeline and keep every stage."""

    if strategy is None:
        strategy = ActivationStrategy()
    rows, was_vector = normalize_inputs(inputs)
    if not rows:
        return ComputeTrace(was_vector=was_vector, exact=exact, dtype=dtype)
    in_features = len(rows[0])

    weights, row_ranges, _ = assemble_output_weight_matrix(
        package, in_features, exact=exact
    )
    z = matmul(rows, weights, exact=exact, chunk_size=chunk_size or strategy.chunk_size)
    add_bias(z, bias, exact)

    trace = ComputeTrace(
        pre_activation=_clone(z),
        weight_matrix=weights,
        row_ranges=row_ranges,
        in_features=in_features,
        was_vector=was_vector,
        exact=exact,
        dtype=dtype,
    )

    # Stage 1: pre-activation saturation (accumulator guard).
    pre_range = strategy.pre_activation_range
    pre_sat = _map_scalar(
        z,
        lambda v: saturate_scalar(
            v,
            pre_range[0] if pre_range else None,
            pre_range[1] if pre_range else None,
            exact=exact,
        ),
    )
    trace.pre_saturation = pre_sat

    # Stage 2: activation.
    activated = _map_scalar(pre_sat, lambda v: activate_scalar(v, strategy, exact=exact))
    trace.activated = activated

    # Stage 3 + 4: per-group post-activation saturation then optional rounding.
    post: list[list[Any]] = [list(row) for row in activated]
    rounded: list[list[Any]] = [list(row) for row in activated]

    for group_index, (start, end) in enumerate(row_ranges):
        group = package.groups[group_index]
        resolved = None
        if strategy.low is not None and strategy.high is not None:
            resolved = (strategy.low, strategy.high)
        else:
            resolved = package.resolved_range_for(group)
        low = resolved[0] if resolved else None
        high = resolved[1] if resolved else None

        for r, row in enumerate(activated):
            for c in range(start, end):
                value = row[c]
                if strategy.clamp_before_round:
                    value = saturate_scalar(value, low, high, exact=exact)
                post[r][c] = value

                if approx_bits or strategy.bits:
                    bits = int(approx_bits or strategy.bits)  # type: ignore[arg-type]
                    if low is not None and high is not None:
                        if exact:
                            value = fake_quantize_exact(
                                value,
                                low,
                                high,
                                bits,
                                rounding=strategy.rounding,
                                symmetric=strategy.symmetric,
                            )
                        else:
                            value = fake_quantize_float(
                                as_float(value),
                                as_float(low),
                                as_float(high),
                                bits,
                                rounding=strategy.rounding,
                            )
                if not strategy.clamp_before_round:
                    value = saturate_scalar(value, low, high, exact=exact)
                rounded[r][c] = value

    trace.post_saturation = post
    trace.rounded = rounded

    output = rounded
    if not exact and dtype:
        output = cast_nested(rounded, dtype)
    trace.output = output
    return trace


def _clone(matrix: list[list[Any]]) -> list[list[Any]]:
    return [list(row) for row in matrix]


def _map_scalar(matrix: list[list[Any]], fn: Any) -> list[list[Any]]:
    return [[fn(value) for value in row] for row in matrix]


def final_output(trace: ComputeTrace) -> Any:
    """Return the final output, squeezing a leading vector input."""

    output = trace.output
    if trace.was_vector and output:
        return list(output[0])
    return output


def floats_of(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [floats_of(v) for v in value]
    return float(value)
