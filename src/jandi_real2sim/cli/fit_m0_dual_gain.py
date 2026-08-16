from __future__ import annotations

import argparse
from pathlib import Path

from jandi_real2sim.cli.common import PROJECT_ROOT
from jandi_real2sim.identification.fit_m0_dual_gain import (
    identify_dual_gain_m0,
    load_dual_gain_fit_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P350/P850 공통 command delay + 조건별 유효 PD M0 식별"
    )
    parser.add_argument(
        "--fit-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "m0_dual_gain.yaml",
    )
    parser.add_argument("--p350-campaign", type=Path, required=True)
    parser.add_argument("--p850-campaign", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "results"
    )
    args = parser.parse_args()
    config = load_dual_gain_fit_config(args.fit_config)
    campaigns = {
        "P350": args.p350_campaign,
        "P850": args.p850_campaign,
    }
    print("Dual-gain M0:")
    print(f"  P350: {args.p350_campaign}")
    print(f"  P850: {args.p850_campaign}")
    print(
        "  delays: "
        + ", ".join(f"{value * 1000.0:.1f} ms" for value in config.delay_values_sec)
    )
    output = identify_dual_gain_m0(campaigns, config, args.output_root)
    print(f"Dual-gain M0 식별 완료: {output}")


if __name__ == "__main__":
    main()

