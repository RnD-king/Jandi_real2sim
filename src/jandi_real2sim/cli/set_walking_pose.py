from __future__ import annotations

import argparse
from pathlib import Path

from jandi_real2sim.dynamixel_bus import DynamixelBus
from jandi_real2sim.experiment import run_trajectory, states_to_pose
from jandi_real2sim.records import timestamped_stem, write_plan_csv
from jandi_real2sim.trajectory import smooth_transition

from .common import PROJECT_ROOT, add_config_argument, load_from_args, require_execute_confirmation


def main() -> None:
    parser = argparse.ArgumentParser(description="현재 자세에서 Jandi 보행 초기자세로 저속 전환")
    add_config_argument(parser)
    parser.add_argument("--transition-sec", type=float)
    parser.add_argument("--execute", action="store_true", help="없으면 하드웨어를 열지 않는 dry-run")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--keep-torque-on", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "plans")
    args = parser.parse_args()
    config = load_from_args(args)
    require_execute_confirmation(args, config)
    duration = args.transition_sec or config.experiment.transition_sec
    target = config.walking_pose()

    if not args.execute:
        # 실제 현재각을 읽을 수 없으므로 영점에서 시작하는 것은 실행 명령이 아니라 계획 확인용이다.
        start = {name: 0.0 for name in target}
        samples = list(smooth_transition(start, target, duration, 100))
        path = args.output_dir / f"{timestamped_stem('walking_pose_dryrun')}.csv"
        count = write_plan_csv(path, samples)
        print("DRY-RUN: 포트 접근, Torque On, Goal Position 전송을 하지 않았습니다.")
        print(f"영점→보행 자세 계획 {count} samples ({duration:.3f} s): {path}")
        return

    with DynamixelBus(config) as bus:
        models = bus.ping_all()
        print(f"Ping OK: {models}")
        current_states = bus.read_state()
        current_pose = states_to_pose(config, current_states)
        samples = smooth_transition(current_pose, target, duration, 100)
        # Torque On 순간 점프를 막기 위해 현재 위치를 먼저 Goal Position으로 쓴다.
        from jandi_real2sim.experiment import pose_to_ticks
        bus.write_goal_ticks(
            pose_to_ticks(config, current_pose, allow_outside_limits=True)
        )
        bus.set_torque(True)
        try:
            count = run_trajectory(
                bus,
                config,
                samples,
                allow_outside_limits=True,
            )
            print(f"보행 초기자세 전환 완료: {count} samples")
        except BaseException:
            bus.set_torque(False)
            raise
        if args.keep_torque_on:
            print("경고: 요청에 따라 Torque On 상태로 포트를 닫습니다.")
        else:
            bus.set_torque(False)
            print("안전을 위해 Torque Off했습니다.")


if __name__ == "__main__":
    main()
