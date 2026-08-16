from __future__ import annotations

import argparse
from pathlib import Path

from jandi_real2sim.cli.common import PROJECT_ROOT
from jandi_real2sim.identification.dataset import load_campaign, load_run
from jandi_real2sim.identification.fit_m0 import identify_m0, load_fit_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL6+LL6 compact-step로 공통 delay/Kp_eff/Kd_eff M0 식별"
    )
    parser.add_argument(
        "--fit-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "m0_ankle_roll.yaml",
        help="M0 모델·탐색 범위 설정",
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        default=PROJECT_ROOT / "configs" / "campaign_20260811_all_joints_A.yaml",
        help="정확한 원본 폴더 이름을 고정한 campaign manifest",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="식별 결과 폴더의 상위 경로",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        help="자동 탐색 대신 사용할 실행 폴더. 정확히 6번 지정",
    )
    args = parser.parse_args()
    config = load_fit_config(args.fit_config)
    if args.run_dir:
        if len(args.run_dir) != 6:
            parser.error("--run-dir은 RL6/LL6 × 반복 1~3, 총 6개여야 합니다.")
        runs = tuple(load_run(path) for path in args.run_dir)
        campaign_id = None
        campaign_source = None
    else:
        campaign = load_campaign(args.campaign)
        runs = tuple(
            run for run in campaign.runs if run.target_joint in config.target_joints
        )
        campaign_id = campaign.campaign_id
        campaign_source = campaign.source
        print(f"Campaign: {campaign.campaign_id} ({campaign.source})")

    print("M0 dataset:")
    for run in runs:
        overrun_count = int(run.overrun_mask.sum())
        print(
            f"  {run.target_joint} repeat={run.repeat_index} "
            f"role={run.split_role} excluded_overrun={overrun_count}: {run.run_dir}"
        )
    output = identify_m0(
        runs,
        config,
        args.output_root,
        campaign_id=campaign_id,
        campaign_source=campaign_source,
    )
    print(f"M0 식별 완료: {output}")


if __name__ == "__main__":
    main()
