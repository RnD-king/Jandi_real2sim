from __future__ import annotations

import copy
import math
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from jandi_real2sim.mode5.canonical_acquisition import AcquisitionStats, TickUnwrapper, _run_delay_samples, timing_statistics
from jandi_real2sim.mode5.canonical_bus import ADDR, CanonicalMode5Bus, State
from jandi_real2sim.mode5.canonical_config import load_canonical_campaign
from jandi_real2sim.mode5.canonical_model import build_model, zoh_command
from jandi_real2sim.mode5.canonical_trajectories import build_dynamic, build_static, command_events
from jandi_real2sim.mode5.canonical_trajectories import static_run_specs
from jandi_real2sim.mode5.cli import _enforce_order
from jandi_real2sim.mode5.spec import CANONICAL_HINGE_AXIS


ROOT = Path(__file__).parents[1]


def resolved_cfg():
    base = load_canonical_campaign(ROOT / "configs/mode5/campaign.yaml")
    hardware = copy.deepcopy(base.hardware)
    hardware.update(serial_device="/dev/null", baudrate=4_000_000, motor_id=1,
                    expected_model_number=321, encoder_zero_raw=2048,
                    expected_homing_offset_raw=0,
                    direction=1, current_direction=1, pwm_direction=1)
    timing = copy.deepcopy(base.timing)
    timing.update(delay_telemetry_target_rate_hz=500, hardware_error_poll_rate_hz=1,
                  severe_overrun_threshold_sec=.1)
    registers = copy.deepcopy(base.registers)
    registers.update(drive_mode=0, position_p_gain=350, position_d_gain=0,
                     bus_watchdog_raw=5, goal_current_raw=500,
                     expected_current_limit_raw=1000, goal_pwm_raw=500,
                     expected_pwm_limit_raw=885)
    safety = copy.deepcopy(base.safety)
    safety.update(software_position_min_rad=-1.2, software_position_max_rad=1.2,
                  maximum_temperature_c=70, minimum_input_voltage_v=10,
                  maximum_input_voltage_v=14, maximum_abs_current_A=3,
                  maximum_abs_pwm_fraction=1, maximum_abs_position_error_rad=2,
                  maximum_consecutive_overruns=100, oscillation_velocity_limit_rad_s=20,
                  transition_duration_sec=0.5, between_runs_sec=1,
                  warmup_procedure="manual")
    geometry = copy.deepcopy(base.geometry)
    geometry.update(arm_lengths_m={"L1": .15, "L2": .25}, arm_mass_kg=.1,
                    arm_com_radius_m=.08, arm_inertia_kg_m2=.001,
                    arm_inertia_reference="about_com", gravity_zero_angle_rad=0,
                    gravity_torque_sign=1)
    loads = copy.deepcopy(base.loads)
    for key, mass in (("m250", .25), ("m500", .5), ("m750", .75)):
        loads[key]["measured_mass_kg"] = mass
    trajectories = copy.deepcopy(base.trajectories)
    trajectories["delay_probe"].update(response_search_sec=.05)
    trajectories["static_calibration"].update(
        static_angles_rad=[-.4, 0, .4], approach_offset_rad=.05,
        approach_duration_sec=.25, inter_point_transfer_duration_sec=.4,
        fixed_settling_hold_sec=.2, minimum_settling_sec=.1,
        averaging_window_sec=.2, maximum_command_speed_rad_s=2.0,
        maximum_settled_abs_velocity_rad_s=.05,
        maximum_settled_position_std_rad=.01, maximum_settled_current_std_A=.02)
    trajectories["slowly_raise_lower"].update(
        center_rad=.1, lower_rad=-.4, upper_rad=.4, speed_rad_s=.2, cycles=1,
        endpoint_hold_sec=.1, center_hold_sec=.1, transition_duration_sec=.5,
        maximum_command_speed_rad_s=2.0)
    trajectories["accelerated_oscillation"].update(
        center_rad=0, amplitude_rad=.2, start_frequency_hz=.2,
        end_frequency_hz=2, duration_sec=2, center_hold_sec=.1,
        transition_duration_sec=.5, maximum_command_speed_rad_s=4)
    trajectories["slow_plus_highfreq"].update(
        center_rad=0, slow_amplitude_rad=.2, slow_frequency_hz=.2,
        high_frequency_amplitude_rad=.02, high_frequency_hz=3,
        duration_sec=2, center_hold_sec=.1, transition_duration_sec=.5,
        maximum_command_speed_rad_s=4)
    return replace(base, campaign={**base.campaign, "id": "test"}, hardware=hardware,
                   timing=timing, registers=registers, safety=safety, geometry=geometry,
                   loads=loads, trajectories=trajectories,
                   approval={"pilot_passed": True, "warmup_acknowledged_at": "2026-01-01T00:00:00Z"},
                   holdout_configuration="L2_m750", execution_order="grouped", randomization_seed=7)


class TickContractTest(unittest.TestCase):
    def test_32768_wrap(self):
        unwrap = TickUnwrapper()
        self.assertEqual([unwrap.update(x) for x in (32766, 32767, 0, 1)], [32766, 32767, 32768, 32769])

    def test_multiple_wraps_and_small_backward_jitter(self):
        unwrap = TickUnwrapper()
        raw = [32765, 32766, 32767, 0, 1, 2, 32766, 32767, 0, 1]
        out = [unwrap.update(x) for x in raw]
        self.assertTrue(all(b >= a for a, b in zip(out, out[1:])))
        jitter = TickUnwrapper()
        self.assertEqual([jitter.update(x) for x in (100, 99, 101)], [100, 99, 101])


class CommandAndTrajectoryContractTest(unittest.TestCase):
    def setUp(self):
        self.cfg = resolved_cfg()

    def test_irregular_delayed_zoh(self):
        event_t = np.array([0.0, .13, .47])
        goal = np.array([1., 2., 3.])
        query = np.array([-.1, 0, .129, .13, .469, .47, .9])
        np.testing.assert_array_equal(zoh_command(query, event_t, goal), [1, 1, 1, 2, 2, 3, 3])

    def test_static_continuity_and_approach_direction(self):
        for approach, expected_sign in (("approach_positive", 1), ("approach_negative", -1)):
            samples = build_static(self.cfg, approach)
            speed = max(abs(b.goal_position_rad-a.goal_position_rad)*self.cfg.target_generation_rate_hz for a,b in zip(samples,samples[1:]))
            self.assertLessEqual(speed, 2.0 + 1e-12)
            phases = {s.phase for s in samples}
            self.assertTrue(any("inter_point_transfer" in phase for phase in phases))
            for index, sample in enumerate(samples):
                if sample.phase.endswith("_approach") and index:
                    delta = sample.goal_position_rad - samples[index-1].goal_position_rad
                    if abs(delta) > 1e-12:
                        self.assertEqual(int(math.copysign(1, delta)), expected_sign)

    def test_all_dynamic_trajectories_are_safe_and_continuous(self):
        for name in ("accelerated_oscillation", "slow_plus_highfreq", "slowly_raise_lower"):
            samples = build_dynamic(self.cfg, name)
            self.assertAlmostEqual(samples[0].goal_position_rad, float(self.cfg.trajectories[name]["center_rad"]))
            self.assertAlmostEqual(samples[-1].goal_position_rad, float(self.cfg.trajectories[name]["center_rad"]))
            limit = float(self.cfg.trajectories[name]["maximum_command_speed_rad_s"])
            speed = max(abs(b.goal_position_rad-a.goal_position_rad)*self.cfg.target_generation_rate_hz for a,b in zip(samples,samples[1:]))
            self.assertLessEqual(speed, limit + 1e-12)
            for sample in samples:
                self.cfg.rad_to_raw(sample.goal_position_rad)
        slow = build_dynamic(self.cfg, "slowly_raise_lower")
        raise_samples = [s for s in slow if s.phase == "cycle_0_raise"]
        speeds = [abs(b.goal_position_rad-a.goal_position_rad)*self.cfg.target_generation_rate_hz for a,b in zip(raise_samples, raise_samples[1:])]
        self.assertAlmostEqual(float(np.median(speeds)), .2, delta=.005)

    def test_delay_events_are_less_frequent_than_telemetry(self):
        from jandi_real2sim.mode5.canonical_trajectories import Sample
        samples = [Sample(i, i/100, "hold", 0 if i < 50 else .1) for i in range(100)]
        self.assertEqual(len(command_events(samples)), 2)
        self.assertEqual(math.ceil(samples[-1].scheduled_time_sec * 500), 495)

    def test_delay_loop_writes_events_but_polls_at_high_rate(self):
        from jandi_real2sim.mode5.canonical_trajectories import Sample

        samples = [Sample(i, i/50, "a" if i < 5 else "b", 0 if i < 5 else .1)
                   for i in range(10)]

        class Clock:
            now = 0
            def monotonic_ns(self):
                return self.now
            def sleep(self, seconds):
                self.now += round(seconds * 1e9)

        class Writer:
            def __init__(self): self.rows = []
            def writerow(self, row): self.rows.append(row)

        class Bus:
            writes = 0
            reads = 0
            goal_raw = 2048
            def write_goal_rad_no_response(self, goal):
                self.writes += 1
                self.goal_raw = self_cfg.rad_to_raw(goal)
            def read_hardware_error(self): return 0
            def read_state(self):
                self.reads += 1
                return State(self.goal_raw, self.reads % 32768, 0, 0, 0, 0, 0, 2048,
                             0, 2048, 120, 30)

        self_cfg = self.cfg
        clock, bus, writer, safety_writer = Clock(), Bus(), Writer(), Writer()
        with patch("jandi_real2sim.mode5.canonical_acquisition.time.monotonic_ns", clock.monotonic_ns), \
             patch("jandi_real2sim.mode5.canonical_acquisition.time.sleep", clock.sleep):
            stats = _run_delay_samples(bus, self.cfg, samples, writer, safety_writer)
        self.assertEqual(bus.writes, 2)
        self.assertEqual(bus.reads, 100)
        self.assertEqual(stats.sample_count, 100)
        self.assertEqual(len(stats.command_tx_after_ns), 2)


class RatesWatchdogAndAxisContractTest(unittest.TestCase):
    def test_official_mx106r_2_control_table_addresses(self):
        expected = {
            "pwm_limit_raw": (36, 2, False), "current_limit_raw": (38, 2, False),
            "bus_watchdog_raw": (98, 1, True), "goal_pwm_raw": (100, 2, True),
            "goal_current_raw": (102, 2, True), "goal_position_raw": (116, 4, True),
        }
        for name, contract in expected.items():
            self.assertEqual(ADDR[name], contract)

    def test_planned_order_is_enforced_and_override_reason_returned(self):
        cfg = resolved_cfg()
        specs = static_run_specs(cfg)
        with self.assertRaises(SystemExit):
            _enforce_order(cfg, SimpleNamespace(override_order=False, override_reason=""),
                           "static", specs[1].relative_directory)
        reason = _enforce_order(cfg, SimpleNamespace(override_order=True, override_reason="fixture repair"),
                                "static", specs[1].relative_directory)
        self.assertEqual(reason, "fixture repair")

    def test_command_and_state_rates_are_independent(self):
        stats = AcquisitionStats(5, (0, 10_000_000, 20_000_000),
                                 (0, 2_000_000, 4_000_000, 6_000_000, 8_000_000),
                                 (0, 2, 0, 4, 0))
        result = timing_statistics(stats)
        self.assertEqual(result["measured_command_rate_hz"], 100)
        self.assertEqual(result["measured_state_rate_hz"], 500)
        self.assertEqual(result["deadline_overrun_count"], 2)
        self.assertEqual(result["deadline_overrun_max_ns"], 4)

    def test_watchdog_lock_and_readback_contract(self):
        cfg = resolved_cfg()
        self.assertNotIn("mode5_registers.bus_watchdog_raw", cfg.execution_missing("collect"))
        self.assertEqual(ADDR["bus_watchdog_raw"], (98, 1, True))
        bus = CanonicalMode5Bus(cfg)
        writes = []
        bus.write = lambda name, value: writes.append((name, value))  # type: ignore[method-assign]
        bus.read = lambda name: 5  # type: ignore[method-assign]
        self.assertEqual(bus.arm_bus_watchdog(), 5)
        self.assertEqual(writes, [("bus_watchdog_raw", 5)])

    def test_mujoco_axis_is_canonical_y(self):
        cfg = resolved_cfg()
        params = {"armature_kg_m2": .01, "coulomb_friction_Nm": .01,
                  "viscous_friction_Nm_s_per_rad": .01}
        model = build_model(cfg, "L1_m250", params, .001)
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "output")
        np.testing.assert_array_equal(model.jnt_axis[joint], CANONICAL_HINGE_AXIS)


if __name__ == "__main__":
    unittest.main()
