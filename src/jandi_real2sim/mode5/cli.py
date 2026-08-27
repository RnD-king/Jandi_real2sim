"""Canonical README-v3 public command-line interface."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .canonical_acquisition import collect_run, run_directory
from .canonical_config import CanonicalCampaign, load_canonical_campaign
from .canonical_trajectories import (
    build_delay, build_dynamic, build_pilot, build_static,
    dynamic_run_specs, static_run_specs,
)
from .spec import CONFIRMATIONS, DEFAULT_CAMPAIGN, DYNAMIC_RUN_COUNT
from .spec import MAIN_TRAJECTORIES, MECHANICAL_CONFIGURATIONS, STATIC_RUN_COUNT


def _base(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_CAMPAIGN)
    return parser


def _execution(parser: argparse.ArgumentParser, kind: str) -> None:
    parser.add_argument("--execute", action="store_true", help="실제 모터에 명령을 전송합니다. 기본값은 dry-run입니다.")
    parser.add_argument("--confirm", default="", help=f"실행 확인 문자열: {CONFIRMATIONS[kind]}")
    parser.add_argument("--resume", action="store_true", help="이미 완료된 valid run은 그대로 두고 건너뜁니다.")
    if kind in ("static", "collect"):
        parser.add_argument("--override-order", action="store_true", help="planned NEXT RUN 순서를 명시적으로 우회합니다.")
        parser.add_argument("--override-reason", default="", help="순서 우회 이유. --override-order와 함께 필수입니다.")


def _load(args: argparse.Namespace) -> CanonicalCampaign:
    return load_canonical_campaign(args.config)


def _require_confirmation(args: argparse.Namespace, kind: str) -> None:
    if args.execute and args.confirm != CONFIRMATIONS[kind]:
        raise SystemExit(f"실기체 실행에는 --execute --confirm {CONFIRMATIONS[kind]}가 필요합니다.")


def _print_missing(cfg: CanonicalCampaign, kind: str) -> bool:
    missing = cfg.execution_missing(kind)
    if not missing:
        print("실기체 실행 필수 설정: resolved")
        return False
    print("실기체 실행 잠금 상태. 미확정 항목:")
    for item in missing:
        print(f"  - {item}")
    return True


def _existing_valid(path: Path) -> bool:
    metadata = path / "metadata.json"
    if not metadata.is_file():
        return False
    try:
        return bool(json.loads(metadata.read_text()).get("valid_flag"))
    except (OSError, ValueError):
        return False


def _next_spec(cfg: CanonicalCampaign, kind: str):
    specs = static_run_specs(cfg) if kind == "static" else dynamic_run_specs(cfg)
    return next((spec for spec in specs if not _existing_valid(run_directory(cfg, spec.relative_directory))), None)


def _enforce_order(cfg: CanonicalCampaign, args: argparse.Namespace, kind: str, relative: str) -> str | None:
    planned = _next_spec(cfg, kind)
    if planned is None or planned.relative_directory == relative:
        return None
    if not args.override_order:
        raise SystemExit(f"요청 run은 planned NEXT RUN이 아닙니다. NEXT: {planned.relative_directory}")
    reason = args.override_reason.strip()
    if not reason:
        raise SystemExit("--override-order에는 비어 있지 않은 --override-reason이 필요합니다.")
    return reason


def _execute_one(cfg: CanonicalCampaign, args: argparse.Namespace, kind: str, relative: str, mechanical: str, trajectory: str, repeat: int, samples: list) -> None:
    if not args.execute:
        print(f"PLAN {relative}: {len(samples)} samples, {len(samples) / cfg.command_rate_hz:.2f} s")
        return
    target = run_directory(cfg, relative)
    if target.exists():
        if args.resume and _existing_valid(target):
            print(f"SKIP valid raw run: {target}")
            return
        raise FileExistsError(f"raw run을 overwrite하지 않습니다: {target}")
    override_reason = _enforce_order(cfg, args, kind, relative) if kind in ("static", "collect") else None
    print(collect_run(cfg, kind, relative, mechanical, trajectory, repeat, samples, override_reason))


def check_main() -> None:
    parser = _base("README-v3 Mode 5 canonical 설정/실험 행렬 검사")
    args = parser.parse_args()
    cfg = _load(args)
    print(f"Campaign: {cfg.campaign_id or '<REQUIRED>'}")
    print(f"Static: {STATIC_RUN_COUNT} sweeps (6 configurations x 2 approaches x 3 repeats)")
    print(f"Dynamic: {DYNAMIC_RUN_COUNT} runs (6 configurations x 3 trajectories x 3 repeats)")
    print("Repeat 1/2/3: 모두 repeatability; repeat 3은 validation 전용이 아님")
    print(f"Holdout: {cfg.holdout_configuration or '<REQUIRED before collection>'}")
    for kind in ("pilot", "static", "delay", "collect"):
        missing = cfg.execution_missing(kind)
        print(f"{kind:7s}: {'READY' if not missing else f'LOCKED ({len(missing)} unresolved)'}")
    _print_missing(cfg, "collect")
    if cfg.campaign_id is not None and cfg.execution_order in ("grouped", "randomized"):
        for kind in ("static", "collect"):
            planned = _next_spec(cfg, kind)
            print(f"NEXT {kind.upper()}: {planned.relative_directory if planned else '<COMPLETE>'}")


def pilot_main() -> None:
    parser = _base("Mode 5 독립 pilot 계획/실행")
    _execution(parser, "pilot")
    args = parser.parse_args()
    _require_confirmation(args, "pilot")
    cfg = _load(args)
    if args.execute and _print_missing(cfg, "pilot"):
        raise SystemExit(2)
    mechanical = str(cfg.pilot.get("mechanical_configuration") or "UNRESOLVED")
    samples = build_pilot(cfg) if not cfg.execution_missing("pilot") else []
    if not samples:
        print("pilot 파형은 null 항목을 채운 뒤 생성됩니다.")
        return
    _execute_one(cfg, args, "pilot", "pilot/run_1", mechanical, "pilot", 1, samples)


def static_main() -> None:
    parser = _base("36개 static sweep의 계획/선택 실행")
    parser.add_argument("--mechanical-configuration", choices=MECHANICAL_CONFIGURATIONS)
    parser.add_argument("--approach", choices=("approach_positive", "approach_negative"))
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3))
    _execution(parser, "static")
    args = parser.parse_args()
    _require_confirmation(args, "static")
    cfg = _load(args)
    if args.execute and (not args.mechanical_configuration or not args.approach or not args.repeat):
        raise SystemExit("실행 시 --mechanical-configuration, --approach, --repeat를 모두 지정하십시오.")
    if args.execute and _print_missing(cfg, "static"):
        raise SystemExit(2)
    specs = [spec for spec in static_run_specs(cfg) if (args.mechanical_configuration is None or spec.mechanical_configuration == args.mechanical_configuration) and (args.approach is None or spec.approach_direction == args.approach) and (args.repeat is None or spec.repeat == args.repeat)]
    if cfg.execution_missing("static"):
        print(f"Static structural plan: {len(specs)} / {STATIC_RUN_COUNT} sweeps; 파형 수치는 아직 null")
        _print_missing(cfg, "static")
        return
    for spec in specs:
        assert spec.approach_direction is not None
        _execute_one(cfg, args, "static", spec.relative_directory, spec.mechanical_configuration, spec.trajectory, spec.repeat, build_static(cfg, spec.approach_direction))
        if args.execute:
            time.sleep(float(cfg.safety["between_runs_sec"]))


def delay_main() -> None:
    parser = _base("Goal Position 전송→Present Current onset delay 계획/실행")
    _execution(parser, "delay")
    args = parser.parse_args()
    _require_confirmation(args, "delay")
    cfg = _load(args)
    if args.execute and _print_missing(cfg, "delay"):
        raise SystemExit(2)
    if cfg.execution_missing("delay"):
        _print_missing(cfg, "delay")
        return
    mechanical = str(cfg.trajectories["delay_probe"]["mechanical_configuration"])
    _execute_one(cfg, args, "delay", "delay/probe_1", mechanical, "delay_probe", 1, build_delay(cfg))


def collect_main() -> None:
    parser = _base("54개 canonical dynamic run 계획/선택 실행")
    parser.add_argument("--mechanical-configuration", choices=MECHANICAL_CONFIGURATIONS)
    parser.add_argument("--trajectory", choices=MAIN_TRAJECTORIES)
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3))
    _execution(parser, "collect")
    args = parser.parse_args()
    _require_confirmation(args, "collect")
    cfg = _load(args)
    if args.execute and (not args.mechanical_configuration or not args.trajectory or not args.repeat):
        raise SystemExit("실행 시 mechanical configuration/trajectory/repeat를 한 run씩 모두 지정하십시오.")
    if args.execute and _print_missing(cfg, "collect"):
        raise SystemExit(2)
    specs = [spec for spec in dynamic_run_specs(cfg) if (args.mechanical_configuration is None or spec.mechanical_configuration == args.mechanical_configuration) and (args.trajectory is None or spec.trajectory == args.trajectory) and (args.repeat is None or spec.repeat == args.repeat)]
    if cfg.execution_missing("collect"):
        print(f"Dynamic structural plan: {len(specs)} / {DYNAMIC_RUN_COUNT} runs; 파형 수치는 아직 null")
        _print_missing(cfg, "collect")
        return
    for spec in specs:
        _execute_one(cfg, args, "collect", spec.relative_directory, spec.mechanical_configuration, spec.trajectory, spec.repeat, build_dynamic(cfg, spec.trajectory))
        if args.execute:
            time.sleep(float(cfg.safety["between_runs_sec"]))


def fit_main() -> None:
    from .canonical_analysis import fit
    parser = _base("static/delay prior + Stage C/D 및 선택적 constrained Stage E 식별")
    parser.add_argument("--fit-config", type=Path, default=DEFAULT_CAMPAIGN.parent / "fit.yaml")
    args = parser.parse_args()
    print(fit(_load(args), args.fit_config))


def validate_main() -> None:
    from .canonical_analysis import validate
    parser = _base("사전 지정 dynamic trajectory holdout configuration의 9-run 검증")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    print(validate(_load(args), args.result))


def report_main() -> None:
    from .canonical_analysis import report
    parser = _base("Mode-5 M1 dynamic holdout plots/metrics/final report 생성")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    print(report(_load(args), args.result))
