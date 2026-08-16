from __future__ import annotations

import argparse

from jandi_real2sim.config import MUJOCO_DOF_ORDER, load_robot_config
from jandi_real2sim.trajectory import multisine_joint

from .measurement_common import (
    add_measurement_arguments,
    collect_or_plan,
    load_pose,
    max_command_speed,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="정책 주파수 대역 기반 단일관절 multisine 동특성 식별",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_measurement_arguments(parser)
    parser.add_argument("joint", choices=MUJOCO_DOF_ORDER)
    parser.add_argument("--amplitude-rad", type=float, required=True)
    parser.add_argument(
        "--frequencies-hz", type=float, nargs="+", required=True,
        help="정책 command PSD에서 선택한 4~6개 주파수",
    )
    parser.add_argument("--duration-sec", type=float, default=20.0)
    parser.add_argument("--fade-sec", type=float, default=1.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--split-role",
        choices=("fit", "validation", "cross_side"),
        required=True,
    )
    parser.add_argument(
        "--max-command-speed-rad-s",
        type=float,
        required=True,
        help="정책에서 확인한 허용 명령속도; 초과 trajectory는 실행 전 차단",
    )
    args = parser.parse_args()
    config = load_robot_config(args.config)
    pose = load_pose(config, args.pose_json)
    frequencies = tuple(args.frequencies_hz)
    if not 4 <= len(frequencies) <= 6:
        raise SystemExit("multisine 주파수는 실험계획대로 4~6개를 지정하세요.")
    min_duration = 5.0 / min(frequencies)
    if args.duration_sec < min_duration:
        raise SystemExit(
            f"duration-sec는 최저주파수 5주기인 {min_duration:.3f}초 이상이어야 합니다."
        )
    samples = list(
        multisine_joint(
            pose,
            args.joint,
            args.amplitude_rad,
            frequencies,
            args.duration_sec,
            args.seed,
            config.bus.command_rate_hz,
            fade_sec=args.fade_sec,
        )
    )
    measured_speed = max_command_speed(
        samples, config.bus.command_rate_hz, args.joint
    )
    if measured_speed > args.max_command_speed_rad_s + 1e-12:
        raise SystemExit(
            f"multisine 최대 명령속도 {measured_speed:.6f} rad/s가 허용값 "
            f"{args.max_command_speed_rad_s:.6f} rad/s를 넘습니다."
        )
    collect_or_plan(
        args,
        config,
        experiment_type="multisine",
        samples=samples,
        center_pose=pose,
        metadata={
            "joint": args.joint,
            "amplitude_rad": args.amplitude_rad,
            "frequencies_hz": list(frequencies),
            "duration_sec": args.duration_sec,
            "fade_sec": args.fade_sec,
            "seed": args.seed,
            "split_role": args.split_role,
            "max_command_speed_rad_s": measured_speed,
            "command_speed_limit_rad_s": args.max_command_speed_rad_s,
        },
        name_suffix=f"{args.joint}_seed{args.seed}_{args.split_role}",
    )


if __name__ == "__main__":
    main()
