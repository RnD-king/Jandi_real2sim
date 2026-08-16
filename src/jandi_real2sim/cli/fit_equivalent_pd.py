from __future__ import annotations

import argparse
from pathlib import Path

from jandi_real2sim.cli.common import PROJECT_ROOT
from jandi_real2sim.identification.fit_equivalent_pd import (
    identify_equivalent_pd,
    load_equivalent_fit_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P350/P850 등가 PD + 공통 지연/백래시/Coulomb 식별"
    )
    parser.add_argument("--fit-config", type=Path, default=PROJECT_ROOT / "configs" / "equivalent_pd.yaml")
    parser.add_argument("--p350-campaign", type=Path, required=True)
    parser.add_argument("--p850-campaign", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    config = load_equivalent_fit_config(args.fit_config)
    print("Equivalent PD + backlash + Coulomb:")
    print(f"  P350: {args.p350_campaign}")
    print(f"  P850: {args.p850_campaign}")
    output = identify_equivalent_pd(
        {"P350": args.p350_campaign, "P850": args.p850_campaign},
        config,
        args.output_root,
    )
    print(f"등가 액추에이터 식별 완료: {output}")


if __name__ == "__main__":
    main()
