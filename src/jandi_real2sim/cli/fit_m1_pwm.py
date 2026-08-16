from __future__ import annotations

import argparse
from pathlib import Path

from jandi_real2sim.cli.common import PROJECT_ROOT
from jandi_real2sim.identification.fit_m1_pwm import (
    identify_pwm_m1,
    load_pwm_m1_fit_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "P350/P850의 실측 Present PWM을 입력으로 공통 출력축 M1 "
            "(drive gain + armature + Coulomb + viscous)을 식별"
        )
    )
    parser.add_argument(
        "--fit-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "m1_pwm.yaml",
    )
    parser.add_argument("--p350-campaign", type=Path, required=True)
    parser.add_argument("--p850-campaign", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    config = load_pwm_m1_fit_config(args.fit_config)
    campaigns = {"P350": args.p350_campaign, "P850": args.p850_campaign}
    print("Measured-PWM M1:")
    print(f"  P350: {args.p350_campaign}")
    print(f"  P850: {args.p850_campaign}")
    print(f"  starts: {len(config.initial_starts)}")
    print("  fit: repeat 1·2 / validation: repeat 3")
    output = identify_pwm_m1(campaigns, config, args.output_root)
    print(f"Measured-PWM M1 식별 완료: {output}")


if __name__ == "__main__":
    main()
