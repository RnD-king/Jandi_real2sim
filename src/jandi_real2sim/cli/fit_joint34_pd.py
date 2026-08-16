from __future__ import annotations

import argparse
from pathlib import Path

from jandi_real2sim.cli.common import PROJECT_ROOT
from jandi_real2sim.identification.fit_joint34_pd import (
    identify_joint34_pd,
    load_joint34_fit_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "P350의 RL/LL 3번·4번만 좌우 대칭 등가 PD로 식별하고 "
            "P850에서 검증"
        )
    )
    parser.add_argument(
        "--fit-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "joint34_pd.yaml",
    )
    parser.add_argument("--p350-campaign", type=Path, required=True)
    parser.add_argument("--p850-campaign", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    args = parser.parse_args()
    config = load_joint34_fit_config(args.fit_config)
    print("Joint-group equivalent PD identification:")
    print(f"  fit P350: {args.p350_campaign}")
    print(f"  held-out P850: {args.p850_campaign}")
    print("  optimize: joint3 Kp/Kd, joint4 Kp/Kd only")
    print("  fixed: joints 1/2/5/6, delay, backlash, friction, encoder tick")
    output = identify_joint34_pd(
        {
            "P350": args.p350_campaign,
            "P850": args.p850_campaign,
        },
        config,
        args.output_root,
    )
    print(f"3·4번 관절군 PD 식별 완료: {output}")


if __name__ == "__main__":
    main()
