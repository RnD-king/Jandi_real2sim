from __future__ import annotations

import unittest
from pathlib import Path

from jandi_real2sim.mode5.canonical_config import load_canonical_campaign
from jandi_real2sim.mode5.canonical_trajectories import dynamic_run_specs, static_run_specs
from jandi_real2sim.mode5.spec import CONFIRMATIONS, DYNAMIC_RUN_COUNT, STATIC_RUN_COUNT


ROOT = Path(__file__).parents[1]


class CanonicalMode5ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_canonical_campaign(ROOT / "configs/mode5/campaign.yaml")

    def test_default_is_deliberately_locked_without_guesses(self) -> None:
        missing = self.cfg.execution_missing("collect")
        self.assertIn("hardware.motor_id", missing)
        self.assertIn("mode5_registers.position_p_gain", missing)
        self.assertIn("bench.geometry.arm_lengths_m.L1", missing)
        self.assertIn("campaign.holdout_configuration", missing)

    def test_exact_static_matrix(self) -> None:
        specs = static_run_specs(self.cfg)
        self.assertEqual(len(specs), STATIC_RUN_COUNT)
        self.assertEqual(STATIC_RUN_COUNT, 36)

    def test_exact_dynamic_matrix_and_repeat_roles(self) -> None:
        specs = dynamic_run_specs(self.cfg)
        self.assertEqual(len(specs), DYNAMIC_RUN_COUNT)
        self.assertEqual(DYNAMIC_RUN_COUNT, 54)
        self.assertEqual({spec.repeat for spec in specs}, {1, 2, 3})

    def test_confirmation_strings_have_one_canonical_source(self) -> None:
        self.assertEqual(
            CONFIRMATIONS,
            {
                "pilot": "PILOT_MX106_MODE5",
                "static": "CALIBRATE_MX106_MODE5",
                "delay": "CALIBRATE_DELAY_MX106_MODE5",
                "collect": "COLLECT_MX106_MODE5",
            },
        )

    def test_canonical_source_tree_has_all_components(self) -> None:
        expected = {
            "README", "campaign", "hardware", "controller", "safety", "bench.geometry", "bench.loads",
            "trajectory.accelerated_oscillation", "trajectory.slow_plus_highfreq",
            "trajectory.slowly_raise_lower", "trajectory.static_calibration", "trajectory.delay_probe",
        }
        self.assertEqual(set(self.cfg.source_files), expected)


if __name__ == "__main__":
    unittest.main()
