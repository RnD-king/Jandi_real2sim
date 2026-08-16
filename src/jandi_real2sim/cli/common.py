from __future__ import annotations

import argparse
from pathlib import Path

from jandi_real2sim.config import RobotConfig, load_robot_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "jandi_mx106.yaml"


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    # 생략하면 프로젝트의 실제 Jandi ID/영점/방향/관절 한계 설정을 사용한다.
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="로봇·통신 설정 YAML 경로",
    )


def load_from_args(args: argparse.Namespace) -> RobotConfig:
    return load_robot_config(args.config)


def require_execute_confirmation(args: argparse.Namespace, config: RobotConfig) -> None:
    if not args.execute:
        return
    try:
        config.require_hardware_ready()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.confirm != "MOVE_JANDI":
        raise SystemExit(
            "실기체 실행에는 --execute --confirm MOVE_JANDI를 함께 입력해야 합니다."
        )
