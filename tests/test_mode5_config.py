from __future__ import annotations

import unittest
from pathlib import Path

from jandi_real2sim.mode5.config import Bench, CONDITIONS, TRAJECTORIES, load_campaign
from jandi_real2sim.mode5.collector import _config_snapshot
from jandi_real2sim.mode5.mujoco_model import build_model
from jandi_real2sim.mode5.trajectories import build, pilot_step


ROOT = Path(__file__).parents[1]


class Mode5ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_campaign(ROOT / "configs" / "mode5" / "campaign.yaml")

    def test_default_is_intentionally_locked(self) -> None:
        unresolved = self.cfg.unresolved_for_execution()
        self.assertIn("conditions.loaded.added_load_mass_kg", unresolved)
        self.assertIn("safety.pilot_approved", unresolved)
        self.assertNotIn(
            "safety.pilot_approved",
            self.cfg.unresolved_for_execution(require_pilot_approval=False),
        )

    def test_pilot_is_small_symmetric_step(self) -> None:
        samples = pilot_step(self.cfg)
        goals = [sample.goal_rad for sample in samples]
        self.assertAlmostEqual(max(goals), self.cfg.hardware.center_rad + 0.035)
        self.assertAlmostEqual(min(goals), self.cfg.hardware.center_rad - 0.035)

    def test_exact_experiment_matrix(self) -> None:
        self.assertEqual(CONDITIONS, ("no_load", "loaded"))
        self.assertEqual(TRAJECTORIES, ("step", "triangle", "sine"))
        self.assertEqual(self.cfg.repeats, (1, 2, 3))
        self.assertEqual(len(CONDITIONS) * len(TRAJECTORIES) * len(self.cfg.repeats), 18)

    def test_component_sources_are_explicit(self) -> None:
        self.assertEqual(
            set(self.cfg.source_files),
            {
                "campaign",
                "hardware",
                "controller",
                "condition.no_load",
                "condition.loaded",
                "trajectory.step",
                "trajectory.triangle",
                "trajectory.sine",
            },
        )

    def test_legacy_manifest_redirects_to_split_config(self) -> None:
        legacy = load_campaign(ROOT / "configs" / "mode5_campaign.yaml")
        self.assertEqual(legacy.source, self.cfg.source)
        self.assertEqual(legacy.trajectories, self.cfg.trajectories)

    def test_bench_equivalent_properties_are_derived(self) -> None:
        bench = Bench(
            bare_horn=False,
            gravity_zero_offset_rad=0.0,
            arm_mass_kg=0.2,
            arm_com_radius_m=0.1,
            arm_inertia_kg_m2=0.003,
            added_load_mass_kg=0.1,
            added_load_radius_m=0.2,
        )
        self.assertAlmostEqual(bench.equivalent_mass_kg, 0.3)
        self.assertAlmostEqual(bench.equivalent_com_radius_m, 0.4 / 3.0)
        self.assertAlmostEqual(bench.equivalent_pivot_inertia_kg_m2, 0.007)

    def test_snapshot_contains_every_source_hash(self) -> None:
        snapshot = _config_snapshot(self.cfg, "no_load", "step")
        self.assertEqual(set(snapshot["sources"]), set(self.cfg.source_files))
        for source in snapshot["sources"].values():
            self.assertEqual(len(source["sha256"]), 64)
        self.assertIn("equivalent_pivot_inertia_kg_m2", snapshot["resolved"]["bench"])

    def test_no_load_is_bare_horn_and_independent_of_loaded_geometry(self) -> None:
        self.assertTrue(self.cfg.benches["no_load"].bare_horn)
        unresolved = self.cfg.unresolved_for_execution(
            condition="no_load", require_pilot_approval=False
        )
        self.assertFalse(any(name.startswith("conditions.loaded") for name in unresolved))
        self.assertFalse(any(name.startswith("conditions.no_load") for name in unresolved))

    def test_all_trajectories_are_contiguous_and_safe(self) -> None:
        for name in TRAJECTORIES:
            samples = build(self.cfg, name)
            self.assertGreater(len(samples), 1)
            self.assertEqual([sample.cycle_index for sample in samples], list(range(len(samples))))
            for sample in samples:
                self.cfg.validate_motion(sample.goal_rad)

    def test_tick_round_trip_is_within_one_tick(self) -> None:
        resolution = 2.0 * 3.141592653589793 / 4096
        for angle in (-0.1, 0.0, 0.1):
            recovered = self.cfg.tick_to_rad(self.cfg.rad_to_tick(angle))
            self.assertLessEqual(abs(recovered - angle), resolution / 2 + 1e-12)

    def test_mujoco_bench_model_builds(self) -> None:
        bench = Bench(
            bare_horn=False,
            gravity_zero_offset_rad=0.0,
            arm_mass_kg=0.2,
            arm_com_radius_m=0.08,
            arm_inertia_kg_m2=0.002,
            added_load_mass_kg=0.1,
            added_load_radius_m=0.15,
        )
        model = build_model(
            bench,
            {
                "armature_kg_m2": 0.005,
                "frictionloss_Nm": 0.02,
                "damping_Nm_s_per_rad": 0.03,
            },
        )
        self.assertEqual(model.nq, 1)
        self.assertEqual(model.nu, 1)

    def test_bare_horn_mujoco_model_builds(self) -> None:
        model = build_model(
            self.cfg.benches["no_load"],
            {
                "armature_kg_m2": 0.005,
                "frictionloss_Nm": 0.02,
                "damping_Nm_s_per_rad": 0.03,
            },
        )
        self.assertEqual(model.nq, 1)


if __name__ == "__main__":
    unittest.main()
