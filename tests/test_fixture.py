import json
from pathlib import Path
import unittest


class FixtureContractTest(unittest.TestCase):
    def test_quantized_fixture_has_per_group_scales_and_bounded_activation(self):
        base = Path(__file__).parents[1]
        payload = json.loads((base / "data/input.json").read_text(encoding="utf-8"))
        scales = [group["scale"] for group in payload["groups"]]
        low, high = payload["activation_range"]
        self.assertEqual(len(scales), 2)
        self.assertNotEqual(scales[0], scales[1])
        self.assertLess(low, high)
        self.assertEqual(len(payload["input"]), 2)

if __name__ == "__main__":
    unittest.main()
