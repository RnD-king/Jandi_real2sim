from __future__ import annotations

import unittest
from pathlib import Path

from jandi_real2sim.config import load_robot_config
from jandi_real2sim.dynamixel_bus import (
    ADDR_POSITION_D_GAIN,
    ADDR_POSITION_I_GAIN,
    ADDR_POSITION_P_GAIN,
    DynamixelBus,
)


class PositionPidWriteTest(unittest.TestCase):
    def test_writes_all_motors_and_verifies_readback(self) -> None:
        root = Path(__file__).parents[1]
        config = load_robot_config(root / "configs" / "jandi_mx106.yaml")
        bus = DynamixelBus(config)
        writes: list[tuple[int, int, int, int]] = []
        bus._write_register = lambda motor_id, address, length, value: writes.append(  # type: ignore[method-assign]
            (motor_id, address, length, value)
        )
        bus.read_actuator_settings = lambda: {  # type: ignore[method-assign]
            motor_id: {
                "position_p_gain": 350,
                "position_i_gain": 0,
                "position_d_gain": 0,
            }
            for motor_id in bus.motor_ids
        }
        settings = bus.write_position_pid_gains(p_gain=350, i_gain=0, d_gain=0)
        self.assertEqual(len(settings), 12)
        self.assertEqual(len(writes), 36)
        expected = {
            (ADDR_POSITION_P_GAIN, 350),
            (ADDR_POSITION_I_GAIN, 0),
            (ADDR_POSITION_D_GAIN, 0),
        }
        for motor_id in bus.motor_ids:
            actual = {(address, value) for mid, address, _, value in writes if mid == motor_id}
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
