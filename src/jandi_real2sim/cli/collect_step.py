from __future__ import annotations

import argparse

from jandi_real2sim.config import MUJOCO_DOF_ORDER, load_robot_config
from jandi_real2sim.trajectory import compact_joint_steps

from .measurement_common import (
    add_measurement_arguments,
    collect_or_plan,
    load_pose,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="다진폭 compact step 액추에이터 식별",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "순서: center 뒤 각 진폭마다 +A,center,-A,center. "
            "repeat 1·2는 fit, repeat 3은 validation으로 별도 폴더에 저장합니다."
        ),
    )
    add_measurement_arguments(parser)
    parser.add_argument("joint", choices=MUJOCO_DOF_ORDER)
    parser.add_argument("--small-amplitude-rad", type=float)
    parser.add_argument("--medium-amplitude-rad", type=float)
    parser.add_argument(
        "--amplitudes-rad",
        type=float,
        nargs="+",
        help="작은 값부터 증가하는 임의 개수 진폭; 기존 small/medium 대신 사용",
    )
    parser.add_argument(
        "--hold-sec",
        type=float,
        help="각 9단계 유지시간; 생략하면 config의 center_hold_sec",
    )
    parser.add_argument("--repeat-index", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    config = load_robot_config(args.config)
    pose = load_pose(config, args.pose_json)
    hold_sec = args.hold_sec or config.experiment.center_hold_sec
    if args.amplitudes_rad is not None:
        if args.small_amplitude_rad is not None or args.medium_amplitude_rad is not None:
            parser.error("--amplitudes-rad와 --small/--medium은 함께 사용할 수 없습니다.")
        amplitudes = tuple(args.amplitudes_rad)
    else:
        if args.small_amplitude_rad is None or args.medium_amplitude_rad is None:
            parser.error("--amplitudes-rad 또는 --small/--medium을 지정해야 합니다.")
        amplitudes = (args.small_amplitude_rad, args.medium_amplitude_rad)
    samples = list(
        compact_joint_steps(
            pose,
            args.joint,
            amplitudes,
            hold_sec,
            config.bus.command_rate_hz,
        )
    )
    role = "fit" if args.repeat_index in (1, 2) else "validation"
    collect_or_plan(
        args,
        config,
        experiment_type="compact_step",
        samples=samples,
        center_pose=pose,
        metadata={
            "joint": args.joint,
            "amplitudes_rad": list(amplitudes),
            "hold_sec": hold_sec,
            "repeat_index": args.repeat_index,
            "split_role": role,
        },
        name_suffix=f"{args.joint}_repeat{args.repeat_index}",
    )


if __name__ == "__main__":
    main()
