from __future__ import annotations

import inspect
import unittest

from jandi_real2sim.mode5 import canonical_acquisition


class ContinuousStateContractTest(unittest.TestCase):
    def test_hardware_error_does_not_replace_state_read(self) -> None:
        source = inspect.getsource(canonical_acquisition._run_samples)
        self.assertLess(source.index("bus.read_state()"), source.index("bus.read_hardware_error()"))
        self.assertNotIn("acquisition_kind", canonical_acquisition.TELEMETRY_FIELDS)

    def test_required_raw_columns(self) -> None:
        fields = set(canonical_acquisition.TELEMETRY_FIELDS)
        for name in (
            "sample_index", "host_time_ns", "command_tx_before_ns", "command_tx_after_ns",
            "goal_position_raw", "present_position_raw", "present_velocity_raw",
            "present_current_raw", "present_pwm_raw", "realtime_tick_unwrapped_ms",
            "current_saturated", "pwm_saturated",
        ):
            self.assertIn(name, fields)


if __name__ == "__main__":
    unittest.main()
