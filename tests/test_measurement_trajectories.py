from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from jandi_real2sim.cli.measurement_common import (
    collect_or_plan,
    max_command_speed,
    validate_samples,
)
from jandi_real2sim.config import load_robot_config
from jandi_real2sim.trajectory import (
    compact_joint_steps,
    hold_pose,
    multisine_joint,
    triangle_joint,
)


class MeasurementTrajectoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).parents[1]
        cls.config = load_robot_config(cls.root / "configs" / "jandi_mx106.yaml")
        cls.pose = cls.config.walking_pose()

    def test_hold_count(self) -> None:
        samples = list(hold_pose(self.pose, 10.0, 100))
        self.assertEqual(len(samples), 1000)
        validate_samples(self.config, samples)

    def test_two_amplitude_step_has_nine_phases(self) -> None:
        samples = list(
            compact_joint_steps(self.pose, "RL6_joint", (0.05, 0.10), 1.5, 100)
        )
        self.assertEqual(len(samples), 1350)
        self.assertEqual(len({sample.phase for sample in samples}), 9)
        validate_samples(self.config, samples)

    def test_triangle_is_bounded_and_returns_to_center(self) -> None:
        samples = list(triangle_joint(self.pose, "RL6_joint", 0.1, 0.5, 5, 100))
        values = [sample.q_cmd_rad["RL6_joint"] for sample in samples]
        self.assertEqual(len(samples), 1001)
        self.assertAlmostEqual(values[0], self.pose["RL6_joint"])
        self.assertAlmostEqual(values[-1], self.pose["RL6_joint"])
        self.assertLessEqual(max(abs(value) for value in values), 0.1 + 1e-12)
        self.assertAlmostEqual(max_command_speed(samples, 100, "RL6_joint"), 0.2)
        validate_samples(self.config, samples)

    def test_multisine_seed_is_reproducible_and_bounded(self) -> None:
        kwargs = dict(
            center=self.pose,
            joint_name="RL6_joint",
            amplitude_rad=0.1,
            frequencies_hz=(0.25, 0.5, 0.75, 1.0),
            duration_s=20.0,
            rate_hz=100,
            fade_sec=1.0,
        )
        first = list(multisine_joint(seed=1, **kwargs))
        again = list(multisine_joint(seed=1, **kwargs))
        other = list(multisine_joint(seed=2, **kwargs))
        values = [sample.q_cmd_rad["RL6_joint"] for sample in first]
        self.assertEqual(values, [sample.q_cmd_rad["RL6_joint"] for sample in again])
        self.assertNotEqual(values, [sample.q_cmd_rad["RL6_joint"] for sample in other])
        self.assertEqual(len(first), 2000)
        self.assertAlmostEqual(values[0], self.pose["RL6_joint"])
        self.assertAlmostEqual(values[-1], self.pose["RL6_joint"])
        self.assertLessEqual(max(abs(value) for value in values), 0.1 + 1e-12)
        validate_samples(self.config, first)

    def test_dry_run_creates_one_directory_with_plan_and_metadata(self) -> None:
        samples = list(hold_pose(self.pose, 0.1, 100))
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                execute=False,
                confirm="",
                pose_id="A",
                output_dir=Path(temp_dir),
                raw_output_dir=Path(temp_dir) / "raw",
            )
            run_dir = collect_or_plan(
                args,
                self.config,
                experiment_type="static_hold",
                samples=samples,
                center_pose=self.pose,
                metadata={"duration_sec": 0.1},
                name_suffix="test",
            )
            self.assertEqual({path.name for path in run_dir.iterdir()}, {"plan.csv", "metadata.json"})
            metadata = json.loads((run_dir / "metadata.json").read_text())
            self.assertEqual(metadata["expected_sample_count"], 10)
            self.assertFalse(metadata["recorded_transition"])


if __name__ == "__main__":
    unittest.main()
