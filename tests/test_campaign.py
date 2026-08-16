from __future__ import annotations

import unittest
from pathlib import Path

from jandi_real2sim.identification.dataset import load_campaign


class NamedCampaignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).parents[1]
        cls.campaign = load_campaign(
            cls.root / "configs" / "campaign_20260811_all_joints_A.yaml"
        )

    def test_exactly_twelve_joints_times_three_repeats(self) -> None:
        self.assertEqual(len(self.campaign.runs), 36)
        keys = {(run.target_joint, run.repeat_index) for run in self.campaign.runs}
        self.assertEqual(len(keys), 36)

    def test_third_repeat_is_validation(self) -> None:
        for run in self.campaign.runs:
            expected = "validation" if run.repeat_index == 3 else "fit"
            self.assertEqual(run.split_role, expected)

    def test_overruns_are_masked_without_changing_raw_files(self) -> None:
        warning_runs = [run for run in self.campaign.runs if run.overrun_mask.any()]
        self.assertEqual(len(warning_runs), 5)
        self.assertEqual(sum(int(run.overrun_mask.sum()) for run in warning_runs), 22)


if __name__ == "__main__":
    unittest.main()
