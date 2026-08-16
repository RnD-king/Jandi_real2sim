from __future__ import annotations

import unittest
from pathlib import Path

from jandi_real2sim.config import load_robot_config
from jandi_real2sim.experiment import (
    HARDWARE_ERROR_SAMPLE,
    STATE_SAMPLE,
    acquisition_kind,
)


class AcquisitionScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_robot_config(
            Path(__file__).parents[1] / "configs" / "jandi_mx106.yaml"
        )

    def test_each_second_has_99_state_and_1_error_slot(self) -> None:
        for second in range(10):
            kinds = [
                acquisition_kind(cycle, self.config)
                for cycle in range(second * 100, (second + 1) * 100)
            ]
            self.assertEqual(kinds.count(STATE_SAMPLE), 99)
            self.assertEqual(kinds.count(HARDWARE_ERROR_SAMPLE), 1)
            self.assertEqual(kinds[-1], HARDWARE_ERROR_SAMPLE)


if __name__ == "__main__":
    unittest.main()
