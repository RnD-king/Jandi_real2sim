from __future__ import annotations

import argparse
import time
from pathlib import Path

from .collector import collect, write_plan
from .config import CONDITIONS, TRAJECTORIES, load_campaign
from .trajectories import build


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT = PROJECT_ROOT / "configs" / "mode5" / "campaign.yaml"


def _config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT)


def check_main() -> None:
    parser = argparse.ArgumentParser(description="Mode 5 최소 실험 설정 정적 검증")
    _config(parser)
    args = parser.parse_args()
    cfg = load_campaign(args.config)
    print(f"Campaign: {cfg.campaign_id}")
    total = 0.0
    for name in TRAJECTORIES:
        samples = build(cfg, name)
        duration = len(samples) / cfg.timing.command_rate_hz
        total += duration * len(CONDITIONS) * len(cfg.repeats)
        print(f"{name:8s}: {len(samples):6d} samples, {duration:8.2f} s/run")
    print(f"18 runs 순수 trajectory 시간: {total / 60:.1f} min")
    unresolved = cfg.unresolved_for_execution()
    if unresolved:
        print("실기체 실행 잠금 상태. 미확정 항목:")
        for name in unresolved:
            print(f"  - {name}")
    else:
        print("정적 설정 완료. 그래도 pilot 확인 후에만 실행하십시오.")


def collect_main() -> None:
    parser = argparse.ArgumentParser(description="Mode 5 no-load/loaded 실험 수집")
    _config(parser)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--trajectory", choices=(*TRAJECTORIES, "all"), default="all")
    parser.add_argument("--repeat", choices=(1, 2, 3), type=int)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="본 실험 전 no-load ±pilot_amplitude 1회만 실행",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    cfg = load_campaign(args.config)
    expected_confirmation = "PILOT_MX106_MODE5" if args.pilot else "MOVE_MX106_MODE5"
    if args.execute and args.confirm != expected_confirmation:
        raise SystemExit(
            f"실기체 실행에는 --execute --confirm {expected_confirmation}가 필요합니다."
        )
    if args.pilot:
        if args.condition != "no_load":
            raise SystemExit("pilot은 no_load에서만 실행합니다.")
        path = (
            collect(cfg, "no_load", "step", 1, pilot=True)
            if args.execute
            else write_plan(cfg, "no_load", "step", 1, pilot=True)
        )
        print(path)
        return
    trajectories = TRAJECTORIES if args.trajectory == "all" else (args.trajectory,)
    repeats = cfg.repeats if args.repeat is None else (args.repeat,)
    for trajectory in trajectories:
        for repeat in repeats:
            path = (
                collect(cfg, args.condition, trajectory, repeat)
                if args.execute
                else write_plan(cfg, args.condition, trajectory, repeat)
            )
            print(path)
            if args.execute:
                time.sleep(cfg.safety.between_runs_sec)


def fit_main() -> None:
    from .analysis import fit

    parser = argparse.ArgumentParser(description="Mode 5 최소 파라미터 계산")
    _config(parser)
    args = parser.parse_args()
    output = fit(load_campaign(args.config))
    print(f"식별 완료: {output}")


def compare_main() -> None:
    from .analysis import compare

    parser = argparse.ArgumentParser(description="18개 Mode 5 실측 그래프 생성")
    _config(parser)
    parser.add_argument(
        "--params",
        type=Path,
        required=True,
        help="jandi-r2s-mode5-fit이 만든 params_mode5.yaml의 정확한 경로",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = compare(load_campaign(args.config), args.params, args.output)
    print(f"MuJoCo 실측 비교: {output}")
