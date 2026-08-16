from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml

from jandi_real2sim.cli.collect_campaign import (
    build_spec_jobs,
    resolve_pid,
    validate_spec_and_jobs,
)
from jandi_real2sim.config import MUJOCO_DOF_ORDER, load_robot_config


class CollectionSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).parents[1]
        cls.raw = yaml.safe_load(
            (cls.root / "configs" / "collection_campaign.yaml").read_text()
        )
        cls.config = load_robot_config(cls.root / "configs" / "jandi_mx106.yaml")

    def test_default_spec_resolves_configured_pid_and_all_experiment_jobs(self) -> None:
        pid = resolve_pid(self.raw["position_pid"])
        self.assertEqual(tuple(pid), MUJOCO_DOF_ORDER)
        expected = self.raw["position_pid"]["default"]
        self.assertTrue(all(values == expected for values in pid.values()))
        jobs = build_spec_jobs(self.raw["experiments"])
        self.assertEqual(len(jobs), 111)
        counts = {
            kind: sum(job.experiment_type == kind for job in jobs)
            for kind in ("compact_step", "triangle", "multisine", "static_hold")
        }
        self.assertEqual(
            counts,
            {"compact_step": 36, "triangle": 36, "multisine": 36, "static_hold": 3},
        )
        self.assertEqual(self.raw["safety"]["consecutive_state_samples"], 5)
        self.assertEqual(self.raw["safety"]["cooldown_every_executed_jobs"], 12)
        validate_spec_and_jobs(self.raw, self.config, jobs)

    def test_joint_pid_override(self) -> None:
        raw = copy.deepcopy(self.raw["position_pid"])
        raw["overrides"] = {"RL3_joint": {"p": 400, "i": 2, "d": 3}}
        pid = resolve_pid(raw)
        self.assertEqual(pid["RL3_joint"], {"p": 400, "i": 2, "d": 3})
        self.assertEqual(pid["LL3_joint"], self.raw["position_pid"]["default"])

    def test_triangle_and_multisine_templates_validate_when_enabled(self) -> None:
        raw = copy.deepcopy(self.raw)
        for experiment in raw["experiments"]:
            experiment["enabled"] = experiment["type"] in ("triangle", "multisine")
        jobs = build_spec_jobs(raw["experiments"])
        self.assertEqual(len(jobs), 72)
        self.assertEqual(
            {job.experiment_type for job in jobs}, {"triangle", "multisine"}
        )
        validate_spec_and_jobs(raw, self.config, jobs)


if __name__ == "__main__":
    unittest.main()
