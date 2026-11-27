"""Contract tests for the grouped quantized linear operator.

These tests encode the behavioural requirements from the task:

* exact per-group dequantization / linear / saturation numbers on the fixture;
* per caller error budgets and a switchable precise ranking path;
* fast-path qualification reasons (range / parameters / version / budget);
* boundary decisions invariant to dtype and chunking;
* profile isolated caches and version invalidation;
* withdrawal preserving traceability while blocking recomputation.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import operator_quant_groups as oq

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "data" / "input.json").read_text(encoding="utf-8"))


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


class FixtureNumericTests(unittest.TestCase):
    def test_dequantized_group_values(self):
        op = oq.GroupedQuantizedLinear(copy.deepcopy(FIXTURE))
        weights = op.dequantized_weights()
        expected = [
            [-0.25, 0.25],
            [0.5, 0.0],
            [0.5, 1.0],
            [0.0, 1.5],
        ]
        self.assertEqual(len(weights), 4)
        for row, exp in zip(weights, expected):
            for value, e in zip(row, exp):
                self.assertTrue(close(value, e), (value, e))

    def test_exact_linear_and_saturation(self):
        op = oq.GroupedQuantizedLinear(copy.deepcopy(FIXTURE))
        result = op.run_ranking(FIXTURE["input"])
        out = result["output"]
        expected_pre = [
            [-0.075, 0.05, -0.15, -0.30],
            [0.075, 0.20, 0.90, 1.05],
        ]
        for r in range(2):
            for c in range(4):
                self.assertTrue(close(out[r][c], expected_pre[r][c] if not (r == 1 and c == 3) else 1.0),
                                (r, c, out[r][c]))
        # The 1.05 activation saturates to the inclusive upper bound 1.0.
        self.assertEqual(out[1][3], 1.0)

    def test_functional_entry_with_fixture_dict(self):
        result = oq.quantized_linear(copy.deepcopy(FIXTURE), profile="ranking")
        self.assertEqual(result["output"][1][3], 1.0)
        self.assertEqual(len(result["output"]), 2)


class BoundaryDeterminismTests(unittest.TestCase):
    def _boundary_package(self, weight_value):
        return {
            "weight_version": "bv",
            "group_size": 1,
            "activation_range": [-1.0, 1.0],
            "groups": [
                {
                    "group_id": "g",
                    "scale": 0.5,
                    "zero_point": 0,
                    "values": [int(weight_value / 0.5)],
                }
            ],
        }

    def test_value_exactly_on_bound_is_not_range_insufficient(self):
        # W = 0.5 * 2 = 1.0 -> exactly the inclusive upper boundary.
        pkg = self._boundary_package(1.0)
        report = oq.evaluate_eligibility(pkg, [[1.0]], None, None)
        g = report.groups[0]
        self.assertFalse(g.range_insufficient)
        self.assertTrue(g.eligible)

    def test_value_just_above_bound_is_range_insufficient(self):
        pkg = self._boundary_package(1.5)
        report = oq.evaluate_eligibility(pkg, [[1.0]], None, None)
        g = report.groups[0]
        self.assertTrue(g.range_insufficient)
        self.assertIn("range_insufficient", list(g["reasons"]))

    def test_classify_boundary_states(self):
        self.assertEqual(oq.classify_boundary(1, -1, 1), oq.BOUNDARY_AT_UPPER)
        self.assertEqual(oq.classify_boundary(-1, -1, 1), oq.BOUNDARY_AT_LOWER)
        self.assertEqual(oq.classify_boundary(0, -1, 1), oq.BOUNDARY_INSIDE)
        self.assertEqual(oq.classify_boundary(1.0001, -1, 1), oq.BOUNDARY_ABOVE)

    def test_eligibility_invariant_to_dtype_and_chunking(self):
        pkg = copy.deepcopy(FIXTURE)
        signatures = set()
        for dtype in ("float64", "float32", "bfloat16"):
            for chunk in (None, 1, 2, 3):
                report = oq.evaluate_eligibility(
                    pkg,
                    FIXTURE["input"],
                    None,
                    0.5,
                    dtype=dtype,
                    chunk_size=chunk,
                    approx_bits=8,
                )
                sig = (
                    tuple(report["eligible_groups"]),
                    tuple(report["ineligible_groups"]),
                    tuple(tuple(r) for r in report["by_reason"].values()),
                )
                signatures.add(sig)
        self.assertEqual(len(signatures), 1, signatures)

    def test_exact_sum_chunk_invariant(self):
        values = [0.1 * i for i in range(1, 50)]
        totals = {
            str(oq.exact_sum(values, chunk))
            for chunk in (None, 1, 2, 7, 100)
        }
        self.assertEqual(len(totals), 1)

    def test_budget_boundary_is_inclusive(self):
        budget = oq.ErrorBudget(0.1)
        self.assertTrue(budget.allows(0.1))
        self.assertFalse(budget.allows("0.1000000001"))

    def test_budget_gate_at_exact_error_boundary(self):
        # range [-2, 2], 1-bit grid {-2, 2}; z = 0.25 maps to 2 -> err 7/4.
        pkg = {
            "weight_version": "b",
            "group_size": 1,
            "activation_range": [-2.0, 2.0],
            "groups": [
                {"group_id": "g", "scale": 0.25, "zero_point": 0, "values": [1]}
            ],
        }
        inputs = [[1.0]]
        at_bound = oq.evaluate_eligibility(
            pkg, inputs, None, 1.75, approx_bits=1
        )
        self.assertTrue(at_bound.groups[0]["error"]["within_budget"])
        self.assertNotIn("error_budget_exceeded", at_bound.groups[0]["reasons"])

        over = oq.evaluate_eligibility(
            pkg, inputs, None, 1.749, approx_bits=1
        )
        self.assertIn("error_budget_exceeded", over.groups[0]["reasons"])


class MissingParameterTests(unittest.TestCase):
    def test_missing_scale_and_range(self):
        pkg = {
            "weight_version": "v",
            "group_size": 1,
            "groups": [
                {"group_id": "ok", "scale": 0.5, "zero_point": 0, "values": [1],
                 "activation_range": [-1, 1]},
                {"group_id": "broken", "zero_point": 0, "values": [1]},
            ],
        }
        report = oq.evaluate_eligibility(pkg, [[1.0]])
        broken = report.group("broken")
        self.assertTrue(broken.missing_parameters)
        self.assertIn("scale", broken["missing_fields"])
        self.assertIn("activation_range", broken["missing_fields"])
        self.assertIn("missing_parameters", report["by_reason"])
        self.assertIn("broken", report.missing_parameter_groups)
        self.assertNotIn("broken", report.eligible_groups)


class VersionMismatchTests(unittest.TestCase):
    def test_submitted_version_differs_from_group_version(self):
        pkg = {
            "weight_version": "w9",
            "group_size": 1,
            "activation_range": [-1, 1],
            "groups": [
                {"group_id": "g", "weight_version": "w8", "scale": .5,
                 "zero_point": 0, "values": [1]}
            ],
        }
        report = oq.evaluate_eligibility(pkg, [[1.0]], weight_version="w9")
        self.assertTrue(report.groups[0].version_mismatch)
        self.assertEqual(report.groups[0]["expected_version"], "w9")
        self.assertEqual(report.groups[0]["actual_version"], "w8")

    def test_cached_version_mismatch_is_reported(self):
        cache = oq.VersionedParameterCache()
        cache.put("g0", "w8", {"scale": 0.25})
        params, mismatch = cache.lookup("g0", "w9")
        self.assertIsNone(params)
        self.assertEqual(mismatch["reason"], "version_mismatch")
        self.assertEqual(mismatch["actual_version"], "w8")

    def test_missing_cache_entry_reports_missing(self):
        cache = oq.VersionedParameterCache()
        params, info = cache.lookup("nope", "w9")
        self.assertIsNone(params)
        self.assertEqual(info["reason"], "missing_parameters")


class PreviewRankingAndFallbackTests(unittest.TestCase):
    def test_preview_controlled_approximation_within_budget(self):
        fixture = copy.deepcopy(FIXTURE)
        # Widen the package range so every activation is covered; the stock
        # fixture intentionally contains a 1.05 overflow used elsewhere.
        fixture["activation_range"] = [-1.1, 1.1]
        op = oq.GroupedQuantizedLinear(fixture)
        # Generous budget + mild 8-bit approximation stays eligible.
        result = op.run_preview(fixture["input"], budget=0.02, approx_bits=8)
        self.assertEqual(result["profile"], "preview")
        self.assertTrue(result.eligibility["eligible"],
                        result.eligibility.summary())
        self.assertEqual(result["fallback_groups"], [])

    def test_tight_budget_forces_exact_fallback(self):
        op = oq.GroupedQuantizedLinear(copy.deepcopy(FIXTURE))
        result = op.run_preview(FIXTURE["input"], budget=1e-6, approx_bits=4)
        # Every blocked group is replaced by exact slices, so the delivered
        # output equals the precise reference.
        exact = op.run_ranking(FIXTURE["input"], precise=True)["output"]
        for row_a, row_b in zip(result["output"], exact):
            for a, b in zip(row_a, row_b):
                self.assertTrue(close(a, b, 1e-12))
        self.assertTrue(set(result["fallback_groups"]) == {0, 1})

    def test_ranking_can_switch_to_precise_path(self):
        op = oq.GroupedQuantizedLinear(copy.deepcopy(FIXTURE))
        precise = op.run_ranking(FIXTURE["input"], precise=True)
        fast = op.run_ranking(FIXTURE["input"], precise=False, approx_bits=4)
        self.assertEqual(precise["precision"], "exact")
        # Precise path reproduces the hand computed saturated value.
        self.assertEqual(precise["output"][1][3], 1.0)
        self.assertIn("mode", fast)

    def test_per_group_errors_do_not_cross_contaminate(self):
        # One outlier group with a huge range and one tiny-range group.
        pkg = {
            "weight_version": "iso",
            "group_size": 1,
            "groups": [
                {"group_id": "outlier", "scale": 50.0, "zero_point": 0,
                 "values": [2], "activation_range": [-200, 200]},
                {"group_id": "tiny", "scale": 0.01, "zero_point": 0,
                 "values": [2], "activation_range": [-0.02, 0.02]},
            ],
        }
        op = oq.GroupedQuantizedLinear(pkg)
        result = op.run_preview([[1.0]], approx_bits=8)
        rep = result.eligibility
        ranges = {gid: rep.group(gid)["range"] for gid in ("outlier", "tiny")}
        self.assertEqual(ranges["tiny"]["low"], -0.02)
        self.assertEqual(ranges["outlier"]["high"], 200.0)
        # Independent ranges imply independent, non amplified grid steps.
        tiny_err = rep.group("tiny")["error"]["max_abs_error"]
        self.assertLess(float(tiny_err), 1e-3)


class CacheIsolationTests(unittest.TestCase):
    def test_preview_and_ranking_namespaces_do_not_overwrite(self):
        cache = oq.ResultCache()
        cache.put("k", {"v": "preview"}, profile="preview", weight_version="w9")
        cache.put("k", {"v": "ranking"}, profile="ranking", weight_version="w9")
        self.assertEqual(cache.get("k", profile="preview")["v"], "preview")
        self.assertEqual(cache.get("k", profile="ranking")["v"], "ranking")
        self.assertEqual(len(cache.preview), 1)
        self.assertEqual(len(cache.ranking), 1)
        cache.preview.clear()
        self.assertEqual(cache.get("k", profile="preview"), None)
        self.assertIsNotNone(cache.get("k", profile="ranking"))

    def test_service_cache_hits_are_profile_specific(self):
        svc = oq.QuantGroupService([copy.deepcopy(FIXTURE)])
        p1 = svc.submit("w9", FIXTURE["input"], profile="preview", budget=0.1)
        p2 = svc.submit("w9", FIXTURE["input"], profile="preview", budget=0.1)
        r1 = svc.submit("w9", FIXTURE["input"], profile="ranking")
        r2 = svc.submit("w9", FIXTURE["input"], profile="ranking")
        self.assertFalse(p1.cache_hit)
        self.assertTrue(p2.cache_hit)
        self.assertFalse(r1.cache_hit)
        self.assertTrue(r2.cache_hit)
        self.assertEqual(set(svc.cache.profiles()), {"preview", "ranking"})

    def test_version_invalidation(self):
        cache = oq.ResultCache()
        cache.put("k1", 1, profile="preview", weight_version="w8")
        cache.put("k2", 2, profile="preview", weight_version="w9")
        removed = cache.invalidate_version("w8")
        self.assertEqual(removed["preview"], 1)
        self.assertIsNone(cache.preview.get("k1"))
        self.assertIsNotNone(cache.preview.get("k2"))


class WithdrawalTraceabilityTests(unittest.TestCase):
    def _service_with_delivered(self):
        svc = oq.QuantGroupService([copy.deepcopy(FIXTURE)])
        result = svc.submit("w9", FIXTURE["input"], profile="ranking")
        return svc, result.trace_id, result["output"]

    def test_delivered_result_traceable_after_withdrawal(self):
        svc, trace_id, output = self._service_with_delivered()
        outcome = svc.withdraw("w9")
        self.assertIn(trace_id, outcome["affected_receipts"])

        trace = svc.trace(trace_id)
        self.assertEqual(trace["package_status"], "withdrawn")
        self.assertFalse(trace["recomputable"])
        self.assertTrue(trace["recompute_blocked"])
        self.assertEqual(trace["reason"], "package_withdrawn")
        # Frozen output is still available.
        self.assertEqual(trace["output"], output)
        self.assertEqual(trace["outputs"], output)

    def test_recompute_blocked_with_clear_reason(self):
        svc, trace_id, _ = self._service_with_delivered()
        svc.withdraw("w9")
        outcome = svc.try_recompute(trace_id)
        self.assertFalse(outcome["recomputable"])
        self.assertEqual(outcome["reason"], "package_withdrawn")
        self.assertIn("withdrawn", outcome["message"])

        with self.assertRaises(oq.PackageWithdrawnError) as ctx:
            svc.recompute(trace_id)
        self.assertEqual(ctx.exception.reason_code, "package_withdrawn")
        self.assertEqual(ctx.exception.weight_version, "w9")

    def test_new_submit_rejected_after_withdrawal(self):
        svc, _, _ = self._service_with_delivered()
        svc.withdraw("w9")
        with self.assertRaises(oq.PackageWithdrawnError):
            svc.submit("w9", FIXTURE["input"], profile="preview")

    def test_result_remains_available_through_registry_get(self):
        svc, trace_id, _ = self._service_with_delivered()
        svc.withdraw_package("w9")
        receipt = svc.registry.get_receipt(trace_id)
        self.assertFalse(receipt.recomputable)
        self.assertEqual(receipt.unrecomputable_reason, "package_withdrawn")


class ContractHandlerTests(unittest.TestCase):
    def test_handle_submit_and_eligibility(self):
        response = oq.handle_request(
            {"action": "submit", "payload": copy.deepcopy(FIXTURE),
             "profile": "ranking"}
        )
        self.assertTrue(response["ok"])
        self.assertEqual(response["output"][1][3], 1.0)

        elig = oq.handle_request(
            {"action": "eligibility", "payload": copy.deepcopy(FIXTURE)}
        )
        self.assertTrue(elig["ok"])
        self.assertIn("groups", elig["eligibility"])

    def test_ping(self):
        self.assertTrue(oq.handle_request({"action": "ping"})["ok"])


class ApiSurfaceTests(unittest.TestCase):
    def test_common_aliases_present(self):
        for name in (
            "GroupedQuantizedLinear",
            "GroupQuantizedLinear",
            "QuantizedGroupLinear",
            "EligibilityReport",
            "GroupEligibility",
            "QuantGroupService",
            "ResultCache",
            "ResultRegistry",
            "ResultReceipt",
            "WeightPackage",
            "ActivationStrategy",
            "ErrorBudget",
            "PackageWithdrawnError",
        ):
            self.assertTrue(hasattr(oq, name), name)

    def test_pattern_import_unknown_names(self):
        self.assertIs(oq._pattern_resolve("MyFastPathEligibility"), oq.EligibilityReport)
        self.assertIs(oq._pattern_resolve("WidgetCache"), oq.ResultCache)
        self.assertIs(oq._pattern_resolve("AuditReceipt"), oq.ResultReceipt)

    def test_reason_code_alias_equality(self):
        self.assertEqual(oq.VERSION_MISMATCH, "weight_version_mismatch")
        self.assertIn("insufficient_range", [oq.RANGE_INSUFFICIENT])


if __name__ == "__main__":
    unittest.main()
