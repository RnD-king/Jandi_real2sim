from __future__ import annotations

import unittest
from pathlib import Path

from jandi_real2sim.cli.collect_p350_campaign import build_jobs, validate_jobs
from jandi_real2sim.config import MUJOCO_DOF_ORDER, load_robot_config


class P350CampaignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).parents[1]
        cls.config = load_robot_config(cls.root / "configs" / "jandi_mx106.yaml")
        cls.jobs = build_jobs(
            MUJOCO_DOF_ORDER,
            (1, 2, 3),
            (0.02, 0.04, 0.07, 0.10),
            (0.015, 0.03, 0.05, 0.07),
        )

    def test_repeat_major_sequence_has_36_jobs(self) -> None:
        self.assertEqual(len(self.jobs), 36)
        self.assertEqual([job.repeat_index for job in self.jobs[:12]], [1] * 12)
        self.assertEqual([job.repeat_index for job in self.jobs[12:24]], [2] * 12)
        self.assertEqual([job.repeat_index for job in self.jobs[24:]], [3] * 12)

    def test_joint2_uses_narrow_amplitudes(self) -> None:
        for job in self.jobs:
            expected_max = 0.07 if job.joint in ("RL2_joint", "LL2_joint") else 0.10
            self.assertEqual(max(job.amplitudes_rad), expected_max)

    def test_all_commands_stay_inside_xml_limits(self) -> None:
        validate_jobs(self.config, self.jobs, hold_sec=1.0)


if __name__ == "__main__":
    unittest.main()
