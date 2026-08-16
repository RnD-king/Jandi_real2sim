from __future__ import annotations

import argparse

from jandi_real2sim.config import load_robot_config
from jandi_real2sim.trajectory import hold_pose

from .measurement_common import (
    add_measurement_arguments,
    collect_or_plan,
    load_pose,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="자세별 정적 처짐·반복성·noise floor 측정",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_measurement_arguments(parser)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--repeat-index", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    config = load_robot_config(args.config)
    pose = load_pose(config, args.pose_json)
    samples = list(
        hold_pose(pose, args.duration_sec, config.bus.command_rate_hz)
    )
    collect_or_plan(
        args,
        config,
        experiment_type="static_hold",
        samples=samples,
        center_pose=pose,
        metadata={
            "duration_sec": args.duration_sec,
            "repeat_index": args.repeat_index,
            "split_role": "baseline",
        },
        name_suffix=f"repeat{args.repeat_index}",
    )


if __name__ == "__main__":
    main()
