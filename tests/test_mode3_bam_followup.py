from dataclasses import replace
from pathlib import Path
import unittest

import mujoco
import numpy as np

from jandi_real2sim.mode3_bam.config import DEFAULT_CAMPAIGN, load_campaign
from jandi_real2sim.mode3_bam.mujoco_validation import ReplayRun
from jandi_real2sim.mode3_bam_followup.run_all import (
    MAXIMUM_BACKLASH_LIMIT_VIOLATION_FRACTION,
    _width_variant,
    backlash_bench_xml,
    simulate_followup,
    uniform_derived_velocity,
)


def configured():
    cfg = load_campaign(DEFAULT_CAMPAIGN)
    return replace(
        cfg,
        hardware={**cfg.hardware, "encoder_zero_raw": 2048, "direction": 1},
    )


def synthetic_run(cfg, duration=0.12):
    condition = cfg.condition("mass1_distance1")
    t = np.arange(0.0, duration + 0.005, 0.01)
    goal = 0.10 * np.sin(2.0 * np.pi * t)
    zeros = np.zeros_like(t)
    metadata = {
        "condition": condition.id,
        "trajectory": "sin_sin",
        "repeat": 3,
        "load_mass_kg": condition.mass_kg,
        "axis_to_load_com_m": condition.distance_m,
        "load_intrinsic_inertia_kg_m2": condition.intrinsic_inertia_kg_m2,
        "load_disk_diameter_m": cfg.bench["load"]["disk_diameter_m"],
        "arm_mass_kg": cfg.bench["arm"]["mass_kg"],
        "arm_com_radius_m": cfg.bench["arm"]["com_radius_m"],
        "arm_inertia_kg_m2": cfg.bench["arm"]["inertia_about_pivot_kg_m2"],
    }
    return ReplayRun(
        Path("synthetic"), metadata, t, goal, zeros.copy(), zeros.copy(),
        zeros.copy(), zeros.copy(), np.full_like(t, 12.0),
        np.ones_like(t, dtype=bool),
    )


PARAMETERS = {
    "kt_nm_per_a": 2.2,
    "resistance_ohm": 2.4,
    "armature_kg_m2": 0.026,
    "coulomb_friction_nm": 0.17,
    "viscous_friction_nm_s_per_rad": 0.0,
    "load_friction": 0.15,
}
CONTROLLER = {
    "command_delay_sec": 0.0,
    "pwm_error_gain_per_raw_rad": 0.0049,
    "pwm_velocity_gain_s_per_rad": 0.0,
    "position_p_gain": 850.0,
}


class Mode3BamFollowupTest(unittest.TestCase):
    def test_uniform_derived_velocity_recovers_smooth_sine(self):
        t = np.arange(0.0, 2.0, 0.01)
        q = np.sin(2.0 * np.pi * t)
        for window in (5, 7, 9):
            grid, _, velocity, valid = uniform_derived_velocity(
                t, q, sample_dt_sec=0.01, window_length=window
            )
            expected = 2.0 * np.pi * np.cos(2.0 * np.pi * grid)
            error = velocity[valid] - expected[valid]
            self.assertLess(np.sqrt(np.mean(error * error)), 0.01)

    def test_backlash_xml_uses_total_width_as_symmetric_range(self):
        cfg = configured()
        run = synthetic_run(cfg)
        width = 0.005854
        model = mujoco.MjModel.from_xml_string(
            backlash_bench_xml(cfg, run, 0.026, 0.001, width)
        )
        self.assertEqual(model.nq, 2)
        joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "passive_backlash"
        )
        np.testing.assert_allclose(
            model.jnt_range[joint], [-width / 2.0, width / 2.0]
        )
        self.assertEqual(int(model.jnt_limited[joint]), 1)

    def test_controller_duty_is_held_between_10ms_updates(self):
        cfg = configured()
        trace = simulate_followup(
            cfg,
            synthetic_run(cfg),
            PARAMETERS,
            CONTROLLER,
            controller_dt_sec=0.01,
            physics_timestep_sec=0.001,
        )
        changes = np.flatnonzero(np.abs(np.diff(trace.duty)) > 1e-12) + 1
        self.assertTrue(len(changes) > 0)
        self.assertTrue(all(index % 10 == 0 for index in changes))

    def test_stateful_backlash_trace_stays_within_quality_threshold(self):
        cfg = configured()
        width = 0.005854
        trace = simulate_followup(
            cfg,
            synthetic_run(cfg),
            PARAMETERS,
            CONTROLLER,
            controller_dt_sec=0.001,
            physics_timestep_sec=0.0001,
            backlash_width_rad=width,
            feedback_view="output_side",
        )
        self.assertEqual(trace.variant, "m3_backlash_output_side")
        violation_fraction = (
            trace.maximum_backlash_limit_violation_rad / (0.5 * width)
        )
        self.assertLessEqual(
            violation_fraction, MAXIMUM_BACKLASH_LIMIT_VIOLATION_FRACTION
        )

    def test_backlash_width_variant_is_unambiguous(self):
        self.assertEqual(
            _width_variant("actuator_side", 0.5),
            "m3_backlash_actuator_side_w0p500",
        )


if __name__ == "__main__":
    unittest.main()
