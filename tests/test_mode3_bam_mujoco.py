import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from jandi_real2sim.mode3_bam.config import DEFAULT_CAMPAIGN, load_campaign
from jandi_real2sim.mode3_bam.mujoco_validation import (
    LOADED_CONDITIONS,
    ReplayRun,
    argument_parser,
    bench_xml,
    build_simulation_run,
    load_replay_run,
    rig_properties,
    simulate_mujoco,
)


def configured():
    cfg = load_campaign(DEFAULT_CAMPAIGN)
    return replace(cfg, hardware={**cfg.hardware, "encoder_zero_raw": 2048, "direction": 1})


def synthetic_run(cfg, *, q0=0.0, torque_enable=True):
    condition = cfg.condition("mass1_distance1")
    metadata = {
        "condition": condition.id,
        "trajectory": "sin_time_square",
        "repeat": 3,
        "load_mass_kg": condition.mass_kg,
        "axis_to_load_com_m": condition.distance_m,
        "load_intrinsic_inertia_kg_m2": condition.intrinsic_inertia_kg_m2,
        "load_disk_diameter_m": cfg.bench["load"]["disk_diameter_m"],
        "arm_mass_kg": cfg.bench["arm"]["mass_kg"],
        "arm_com_radius_m": cfg.bench["arm"]["com_radius_m"],
        "arm_inertia_kg_m2": cfg.bench["arm"]["inertia_about_pivot_kg_m2"],
    }
    t = np.linspace(0.0, 0.05, 6)
    zeros = np.zeros_like(t)
    return ReplayRun(
        Path("synthetic"), metadata, t, zeros, np.full_like(t, q0), zeros,
        zeros, zeros, np.full_like(t, 12.0), np.full_like(t, torque_enable, dtype=bool),
    )


class Mode3BamMujocoTest(unittest.TestCase):
    def test_exactly_six_loaded_conditions_are_selectable(self):
        self.assertEqual(len(LOADED_CONDITIONS), 6)
        self.assertNotIn("no_load", LOADED_CONDITIONS)
        self.assertIn("mass1_distance1", LOADED_CONDITIONS)
        self.assertIn("mass3_distance2", LOADED_CONDITIONS)

    def test_cli_defaults_to_all_trajectory_simulation_without_comparison(self):
        args = argument_parser().parse_args(["--condition", "mass1_distance1"])
        self.assertIsNone(args.trajectory)
        self.assertIsNone(args.compare_repeat)
        self.assertEqual(args.voltage, 12.0)

    def test_generated_simulation_run_does_not_load_a_measured_repeat(self):
        cfg = configured()
        run = build_simulation_run(cfg, "mass1_distance1", "sin_sin")
        self.assertEqual(run.metadata["source"], "generated_from_trajectory_yaml")
        self.assertNotIn("repeat", run.metadata)
        np.testing.assert_allclose(run.voltage, 12.0)
        self.assertAlmostEqual(run.q[0], run.goal[0])

    def test_replay_rebuilds_current_from_raw_using_current_axis(self):
        cfg = configured()
        with tempfile.TemporaryDirectory() as temporary:
            cfg = replace(
                cfg,
                project_root=Path(temporary),
                campaign={**cfg.campaign, "output_root": "raw"},
            )
            attempt = (
                cfg.output_root / str(cfg.campaign_id) / "mass1_distance1"
                / "sin_sin" / "repeat_3" / "attempt_001"
            )
            attempt.mkdir(parents=True)
            (attempt / "metadata.json").write_text(json.dumps({
                "valid": True,
                "condition": "mass1_distance1",
                "trajectory": "sin_sin",
                "repeat": 3,
            }))
            (attempt / "telemetry.csv").write_text(
                "host_time_sec,goal_position_rad,present_position_rad,"
                "present_velocity_rad_s,present_pwm_fraction,present_current_raw,"
                "present_current_A,input_voltage_V,torque_enable\n"
                "0.0,0.0,0.0,0.0,0.0,-10,0.0336,12.0,1\n"
                "0.01,0.0,0.0,0.0,0.0,-20,0.0672,12.0,1\n"
            )
            run = load_replay_run(cfg, "mass1_distance1", "sin_sin", 3)
            np.testing.assert_allclose(run.current, [-0.0336, -0.0672])
            self.assertEqual(
                run.metadata["comparison_current_source"], "present_current_raw"
            )

    def test_aggregate_rig_preserves_pivot_inertia(self):
        cfg = configured()
        run = synthetic_run(cfg)
        rig = rig_properties(run.metadata)
        reconstructed = (
            rig.inertia_about_com_kg_m2
            + rig.physical_mass_kg * rig.center_of_mass_m ** 2
        )
        self.assertAlmostEqual(reconstructed, rig.inertia_about_pivot_kg_m2)

    def test_xml_uses_physical_inertia_plus_identified_armature(self):
        cfg = configured(); run = synthetic_run(cfg)
        armature = 0.025
        model = mujoco.MjModel.from_xml_string(bench_xml(cfg, run, armature, 0.001))
        self.assertEqual(model.nq, 1)
        self.assertAlmostEqual(float(model.dof_armature[0]), armature)

    def test_zero_upright_state_remains_zero(self):
        cfg = configured(); run = synthetic_run(cfg)
        parameters = {
            "kt_nm_per_a": 2.2,
            "resistance_ohm": 2.4,
            "armature_kg_m2": 0.026,
            "coulomb_friction_nm": 0.17,
            "viscous_friction_nm_s_per_rad": 0.0,
            "load_friction": 0.15,
        }
        controller = {
            "command_delay_sec": 0.0,
            "pwm_error_gain_per_raw_rad": 0.0049,
            "pwm_velocity_gain_s_per_rad": 0.0,
            "position_p_gain": 850.0,
        }
        trace = simulate_mujoco(cfg, run, "m3", parameters, controller)
        np.testing.assert_allclose(trace.q, 0.0, atol=1e-12)
        np.testing.assert_allclose(trace.current, 0.0, atol=1e-12)

    def test_runtime_frictionloss_holds_an_offset_pendulum(self):
        cfg = configured(); run = synthetic_run(cfg, q0=0.1, torque_enable=False)
        parameters = {
            "kt_nm_per_a": 2.2,
            "resistance_ohm": 2.4,
            "armature_kg_m2": 0.026,
            "coulomb_friction_nm": 10.0,
            "viscous_friction_nm_s_per_rad": 0.0,
        }
        controller = {
            "command_delay_sec": 0.0,
            "pwm_error_gain_per_raw_rad": 0.0049,
            "pwm_velocity_gain_s_per_rad": 0.0,
            "position_p_gain": 850.0,
        }
        held = simulate_mujoco(cfg, run, "m1", parameters, controller)
        free = simulate_mujoco(
            cfg, run, "m1", {**parameters, "coulomb_friction_nm": 0.0}, controller
        )
        held_drift = float(np.max(np.abs(held.q - held.q[0])))
        free_drift = float(np.max(np.abs(free.q - free.q[0])))
        self.assertLess(held_drift, 0.1 * free_drift)
        self.assertGreater(np.max(np.abs(held.friction_torque)), 0.0)


if __name__ == "__main__":
    unittest.main()
