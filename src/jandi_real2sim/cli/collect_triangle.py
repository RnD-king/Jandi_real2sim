from __future__ import annotations

import argparse

from jandi_real2sim.config import MUJOCO_DOF_ORDER, load_robot_config
from jandi_real2sim.trajectory import triangle_joint

from .measurement_common import (
    add_measurement_arguments,
    collect_or_plan,
    load_pose,
    max_command_speed,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="저속 방향전환 triangle 입력으로 마찰·백래시 진단",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_measurement_arguments(parser)
    parser.add_argument("joint", choices=MUJOCO_DOF_ORDER)
    parser.add_argument("--amplitude-rad", type=float, required=True)
    parser.add_argument("--frequency-hz", type=float, required=True)
    parser.add_argument("--cycles", type=int, required=True)
    parser.add_argument(
        "--max-command-speed-rad-s",
        type=float,
        required=True,
        help="정책에서 확인한 허용 명령속도; 초과 trajectory는 실행 전 차단",
    )
    parser.add_argument("--repeat-index", type=int, required=True)
    args = parser.parse_args()
    config = load_robot_config(args.config)
    pose = load_pose(config, args.pose_json)
    samples = list(
        triangle_joint(
            pose,
            args.joint,
            args.amplitude_rad,
            args.frequency_hz,
            args.cycles,
            config.bus.command_rate_hz,
        )
    )
    measured_speed = max_command_speed(
        samples, config.bus.command_rate_hz, args.joint
    )
    if measured_speed > args.max_command_speed_rad_s + 1e-12:
        raise SystemExit(
            f"triangle 최대 명령속도 {measured_speed:.6f} rad/s가 허용값 "
            f"{args.max_command_speed_rad_s:.6f} rad/s를 넘습니다."
        )
    collect_or_plan(
        args,
        config,
        experiment_type="triangle",
        samples=samples,
        center_pose=pose,
        metadata={
            "joint": args.joint,
            "amplitude_rad": args.amplitude_rad,
            "frequency_hz": args.frequency_hz,
            "cycles": args.cycles,
            "duration_sec": args.cycles / args.frequency_hz,
            "max_command_speed_rad_s": measured_speed,
            "command_speed_limit_rad_s": args.max_command_speed_rad_s,
            "repeat_index": args.repeat_index,
            "split_role": "diagnostic",
        },
        name_suffix=f"{args.joint}_repeat{args.repeat_index}",
    )


if __name__ == "__main__":
    main()
