from __future__ import annotations

import unittest
from pathlib import Path

from jandi_real2sim.cli.measurement_common import LiveSafetyMonitor
from jandi_real2sim.config import load_robot_config
from jandi_real2sim.dynamixel_bus import MotorState
from jandi_real2sim.trajectory import TrajectorySample


class LiveSafetyMonitorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).parents[1]
        cls.config = load_robot_config(root / "configs" / "jandi_mx106.yaml")
        cls.pose = cls.config.walking_pose()
        cls.limits = {
            "max_temperature_c": 55,
            "min_input_voltage_v": 9.6,
            "max_abs_current_a": 2.5,
            "max_abs_pwm_percent": 85.0,
            "max_abs_position_error_rad": 0.25,
            "consecutive_state_samples": 5,
        }

    def states(self, **overrides: int) -> dict[int, MotorState]:
        result: dict[int, MotorState] = {}
        for joint in self.config.joints:
            assert joint.motor_id is not None
            values = {
                "present_position_tick": joint.rad_to_tick(self.pose[joint.name]),
                "present_velocity_raw": 0,
                "present_pwm_raw": 0,
                "present_current_raw": 0,
                "position_trajectory_tick": joint.rad_to_tick(self.pose[joint.name]),
                "velocity_trajectory_raw": 0,
                "realtime_tick_ms": 0,
                "input_voltage_raw": 120,
                "temperature_c": 30,
                "moving": 0,
                "moving_status": 0,
            }
            if joint.name == "RL6_joint":
                values.update(overrides)
            result[joint.motor_id] = MotorState(**values)
        return result

    def sample(self) -> TrajectorySample:
        return TrajectorySample(0, 0.0, "test", dict(self.pose))

    def test_state_limit_requires_five_consecutive_samples(self) -> None:
        monitor = LiveSafetyMonitor(self.config, dict(self.limits))
        states = self.states(present_current_raw=800)  # 2.688 A
        for _ in range(4):
            monitor.check(self.sample(), states, None)
        with self.assertRaisesRegex(RuntimeError, "abs_current_a"):
            monitor.check(self.sample(), states, None)

    def test_normal_sample_resets_consecutive_counter(self) -> None:
        monitor = LiveSafetyMonitor(self.config, dict(self.limits))
        bad = self.states(present_pwm_raw=800)  # 90.4 %
        good = self.states()
        for _ in range(4):
            monitor.check(self.sample(), bad, None)
        monitor.check(self.sample(), good, None)
        for _ in range(4):
            monitor.check(self.sample(), bad, None)

    def test_hardware_error_stops_immediately(self) -> None:
        monitor = LiveSafetyMonitor(self.config, dict(self.limits))
        motor_id = int(self.config.joints[0].motor_id)
        with self.assertRaisesRegex(RuntimeError, "Hardware Error"):
            monitor.check(self.sample(), None, {motor_id: 4})


if __name__ == "__main__":
    unittest.main()
