from dataclasses import replace
import unittest
from unittest.mock import patch

from jandi_real2sim.mode3_bam.config import DEFAULT_CAMPAIGN, EDITABLE, load_campaign
import numpy as np

from jandi_real2sim.mode3_bam.analysis import MODEL_PARAMETERS, _friction_budget, _system_inertia, _zoh
from jandi_real2sim.mode3_bam.acquisition import _validate_samples
from jandi_real2sim.mode3_bam.calibrate import (
    capture_upright_zero, engineering_limits, infer_signs, jog_raw_positive,
)
from jandi_real2sim.mode3_bam.trajectories import build, trajectories_for


def configured():
    cfg = load_campaign(DEFAULT_CAMPAIGN)
    hardware = {**cfg.hardware, "encoder_zero_raw": 2048, "direction": 1}
    return replace(cfg, hardware=hardware)


class Mode3BamToolTest(unittest.TestCase):
    def test_seven_conditions_three_repeats_and_fixed_gain(self):
        cfg = load_campaign(DEFAULT_CAMPAIGN)
        self.assertEqual(len(cfg.conditions), 7)
        self.assertEqual(cfg.repetitions, (1, 2, 3))
        self.assertEqual(cfg.fit_repetitions, (1, 2))
        self.assertEqual(cfg.validation_repetitions, (3,))
        self.assertEqual(cfg.registers["operating_mode"], 3)
        self.assertEqual(cfg.registers["position_p_gain"], 850)
        self.assertEqual(EDITABLE["campaign"], ("campaign", "campaign", "id"))

    def test_no_load_and_loaded_batches_have_expected_trajectories(self):
        self.assertEqual(trajectories_for("no_load"), ("delay_probe", "backlash_probe"))
        self.assertEqual(trajectories_for("mass1_distance1"), (
            "sin_time_square", "sin_sin", "up_and_down", "lift_and_drop"
        ))

    def test_measured_bench_constants_and_disk_inertia(self):
        cfg = load_campaign(DEFAULT_CAMPAIGN)
        expected = {
            "mass1": (1, 0.3083, 0.00015673875),
            "mass2": (2, 0.5642, 0.00031347750),
            "mass3": (3, 0.8201, 0.00047021625),
        }
        for mass_key, (count, mass, intrinsic) in expected.items():
            condition = cfg.condition(f"{mass_key}_distance1")
            self.assertEqual(condition.disk_count, count)
            self.assertAlmostEqual(condition.mass_kg, mass)
            self.assertAlmostEqual(condition.distance_m, 0.10)
            self.assertAlmostEqual(condition.intrinsic_inertia_kg_m2, intrinsic)
        self.assertAlmostEqual(cfg.condition("mass1_distance2").distance_m, 0.15)
        self.assertAlmostEqual(cfg.bench["arm"]["inertia_about_pivot_kg_m2"], 0.000404978)
        self.assertAlmostEqual(cfg.bench["no_load_hardware"]["horn_mass_kg"], 0.0039)
        self.assertAlmostEqual(cfg.bench["loaded_mounting_hardware"]["horn_mass_kg"], 0.006)
        self.assertAlmostEqual(cfg.bench["loaded_mounting_hardware"]["horn_fastener_mass_kg"], 0.0063)

    def test_system_inertia_includes_finite_disk_size(self):
        metadata = {
            "load_mass_kg": 0.3083, "axis_to_load_com_m": 0.10,
            "load_intrinsic_inertia_kg_m2": 0.00015673875,
            "arm_inertia_kg_m2": 0.000404978,
        }
        expected = 0.005 + 0.000404978 + 0.3083 * 0.10 ** 2 + 0.00015673875
        self.assertAlmostEqual(_system_inertia(0.005, metadata), expected)

    def test_bam_models_include_directional_m5(self):
        self.assertEqual(tuple(MODEL_PARAMETERS), ("m1", "m2", "m3", "m4", "m5"))
        self.assertIn("motor_load_friction", MODEL_PARAMETERS["m5"])
        self.assertIn("external_load_friction", MODEL_PARAMETERS["m5"])

    def test_m5_collapses_to_m4_when_directional_coefficients_match(self):
        common = {
            "coulomb_friction_nm": 0.1, "viscous_friction_nm_s_per_rad": 0.2,
            "stribeck_friction_nm": 0.3, "stribeck_velocity_rad_s": 0.4,
            "stribeck_alpha": 1.2, "load_friction": 0.25,
            "load_stribeck_friction": 0.15, "motor_load_friction": 0.25,
            "external_load_friction": 0.25, "motor_load_stribeck_friction": 0.15,
            "external_load_stribeck_friction": 0.15,
        }
        m4 = _friction_budget("m4", common, 0.12, 1.1, -0.4)
        m5 = _friction_budget("m5", common, 0.12, 1.1, -0.4)
        self.assertAlmostEqual(m4, m5)

    def test_every_trajectory_builds_and_lift_drop_releases_torque(self):
        cfg = configured()
        names = trajectories_for("no_load") + trajectories_for("mass1_distance1")
        generated = {name: build(cfg, name) for name in names}
        self.assertTrue(all(generated.values()))
        self.assertTrue(any(not sample.torque_enable for sample in generated["lift_and_drop"]))
        self.assertTrue(generated["lift_and_drop"][-1].torque_enable)
        self.assertAlmostEqual(generated["lift_and_drop"][-1].goal_rad, 0.0)
        released = [sample for sample in generated["lift_and_drop"] if sample.phase == "released"]
        self.assertEqual(len(released), round(0.30 * cfg.command_rate_hz))
        self.assertAlmostEqual(cfg.trajectories["lift_and_drop"]["catch_angle_rad"], -0.45)
        self.assertAlmostEqual(cfg.trajectories["lift_and_drop"]["catch_abs_velocity_rad_s"], 3.5)
        self.assertTrue(all(sample.torque_enable for sample in generated["sin_time_square"]))

    def test_delay_uses_previous_value_hold(self):
        source_t = np.asarray([0.0, 1.0, 2.0])
        values = np.asarray([0.0, 10.0, 20.0])
        query = np.asarray([-0.1, 0.5, 1.0, 1.9, 3.0])
        np.testing.assert_array_equal(_zoh(source_t, values, query), [0.0, 0.0, 10.0, 10.0, 20.0])

    def test_goal_preflight_rejects_out_of_range_sample(self):
        cfg = configured()
        cfg = replace(cfg, safety={**cfg.safety, "software_position_min_rad": -0.2,
                                   "software_position_max_rad": 0.2})
        samples = build(cfg, "sin_time_square")
        with self.assertRaisesRegex(ValueError, "software limit"):
            _validate_samples(cfg, samples)

    def test_calibration_infers_common_sign_convention(self):
        self.assertEqual(infer_signs(True, [3, 4, 5], [8, 9, 10]), {
            "direction": 1, "current_direction": 1, "pwm_direction": 1,
            "gravity_torque_sign": 1,
        })
        self.assertEqual(infer_signs(False, [3, 4, 5], [8, 9, 10])["direction"], -1)

    def test_calibration_converts_hardware_limits(self):
        converted = engineering_limits({
            "temperature_limit_c": 80, "min_voltage_limit_raw": 100,
            "max_voltage_limit_raw": 160, "current_limit_raw": 1000,
            "pwm_limit_raw": 885, "velocity_limit_raw": 100,
        })
        self.assertAlmostEqual(converted["hardware_current_limit_A"], 3.36)
        self.assertAlmostEqual(converted["hardware_pwm_limit_fraction"], 1.00005)

    def test_gui_calibration_motion_is_bounded_and_finishes_torque_off(self):
        class FakeState:
            present_position_raw = 2048
            present_current_raw = 5
            present_pwm_raw = 10

        class FakeBus:
            def __init__(self):
                self.torque_commands = []
                self.goals = []

            def torque(self, enabled): self.torque_commands.append(enabled)
            def read_state(self): return FakeState()
            def read_hardware_error(self): return 0
            def read(self, name):
                return {"operating_mode": 3, "position_p_gain": 850,
                        "min_position_limit_raw": 0, "max_position_limit_raw": 4095}[name]
            def write(self, name, value):
                if name == "goal_position_raw": self.goals.append(value)

        bus = FakeBus()
        self.assertEqual(capture_upright_zero(bus), 2048)
        with patch("jandi_real2sim.mode3_bam.calibrate.time.sleep"):
            currents, pwms = jog_raw_positive(bus, 2048, 4)
        self.assertEqual(currents, [5] * 4)
        self.assertEqual(pwms, [10] * 4)
        self.assertEqual(min(bus.goals), 2048)
        self.assertEqual(max(bus.goals), 2052)
        self.assertFalse(bus.torque_commands[-1])


if __name__ == "__main__":
    unittest.main()
