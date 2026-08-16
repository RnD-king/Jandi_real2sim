from __future__ import annotations

import argparse
from pathlib import Path

from jandi_real2sim.cli.common import PROJECT_ROOT
from jandi_real2sim.identification.dataset import load_campaign


def main() -> None:
    parser = argparse.ArgumentParser(
        description="이름으로 고정한 Jandi Real2Sim campaign 36개 run 검증"
    )
    parser.add_argument(
        "campaign",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "configs" / "campaign_20260811_all_joints_A.yaml",
    )
    args = parser.parse_args()
    campaign = load_campaign(args.campaign)
    print(f"Campaign: {campaign.campaign_id}")
    print(f"Manifest: {campaign.source}")
    print(f"Runs: {len(campaign.runs)}")
    warning_runs = 0
    excluded_samples = 0
    for run in campaign.runs:
        overruns = int(run.overrun_mask.sum())
        warning_runs += overruns > 0
        excluded_samples += overruns
        status = "PASS" if overruns == 0 else f"PASS_WITH_MASK overrun={overruns}"
        print(
            f"  {status:26s} {run.target_joint} repeat={run.repeat_index} "
            f"role={run.split_role} {run.run_dir.name}"
        )
    print(
        f"RESULT: PASS, warning_runs={warning_runs}, "
        f"excluded_overrun_samples={excluded_samples}"
    )


if __name__ == "__main__":
    main()
