from __future__ import annotations

import argparse
from pathlib import Path

from jandi_real2sim.cli.common import PROJECT_ROOT
from jandi_real2sim.identification.fit_m1_staged import (
    identify_staged_pwm_m1,
    load_staged_m1_fit_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="실측 PWM M1 단계식 식별: triangle → multisine → 전체 미세조정"
    )
    parser.add_argument(
        "--fit-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "m1_pwm_staged.yaml",
    )
    parser.add_argument("--p350-campaign", type=Path, required=True)
    parser.add_argument("--p850-campaign", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    config = load_staged_m1_fit_config(args.fit_config)
    campaigns = {"P350": args.p350_campaign, "P850": args.p850_campaign}
    print("Staged measured-PWM M1:")
    print(f"  P350: {args.p350_campaign}")
    print(f"  P850: {args.p850_campaign}")
    print("  stage 1: triangle / drive gain + Coulomb")
    print("  stage 2: multisine / drive gain + armature + viscous")
    print("  stage 3: step + triangle + multisine / four-parameter refinement")
    output = identify_staged_pwm_m1(campaigns, config, args.output_root)
    print(f"Staged measured-PWM M1 식별 완료: {output}")


if __name__ == "__main__":
    main()
