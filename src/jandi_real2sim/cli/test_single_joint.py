from __future__ import annotations

import argparse
from pathlib import Path

from jandi_real2sim.config import MUJOCO_DOF_ORDER
from jandi_real2sim.dynamixel_bus import DynamixelBus
from jandi_real2sim.experiment import pose_to_ticks, run_trajectory, states_to_pose
from jandi_real2sim.records import (
    RawCsvRecorder,
    timestamped_stem,
    write_metadata,
    write_plan_csv,
)
from jandi_real2sim.trajectory import compact_joint_step, smooth_transition

from .common import PROJECT_ROOT, add_config_argument, load_from_args, require_execute_confirmation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="보행 초기자세에서 한 관절만 ±step excitation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "시험 순서: 현재 자세에서 보행 자세로 저속 전환한 뒤 "
            "center,+A,center,-A,center를 반복합니다. "
            "초기자세 전환은 기록하지 않으며, step 표본 수 = "
            "hold_sec × 100 Hz × 5단계 × repeats입니다."
        ),
    )
    add_config_argument(parser)

    # 시험 대상 한 관절만 step 명령을 받고 나머지 11개는 보행 기본자세를 유지한다.
    parser.add_argument(
        "joint",
        choices=MUJOCO_DOF_ORDER,
        help="시험할 관절(RL1..RL6 또는 LL1..LL6)",
    )
    # 관절의 walking_rad를 중심으로 +A와 -A를 모두 시험한다. 단위는 rad다.
    parser.add_argument(
        "--amplitude-rad",
        type=float,
        help="보행 기본각 기준 ±step 진폭; 생략하면 config의 기본값",
    )
    # center,+A,center,-A,center 다섯 단계 각각에 동일하게 적용되는 유지시간이다.
    parser.add_argument(
        "--hold-sec",
        type=float,
        help="다섯 step 단계 각각의 유지시간; 생략하면 config의 기본값",
    )
    # 한 repeat는 center,+A,center,-A,center의 완전한 한 묶음이다.
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="5단계 step 묶음 반복 횟수",
    )
    # 이 플래그가 없으면 포트를 열거나 Torque On하지 않고 계획 CSV만 만든다.
    parser.add_argument(
        "--execute",
        action="store_true",
        help="실기체에 명령 실행; 생략 시 안전한 dry-run",
    )
    # --execute의 우발 입력을 막는 두 번째 안전장치로 정확한 문자열을 요구한다.
    parser.add_argument(
        "--confirm",
        default="",
        help="실행 확인 문자열; 실기체 실행 시 MOVE_JANDI 필수",
    )
    # dry-run일 때 사전에 계산한 명령 trajectory와 metadata를 실행별 폴더에 저장한다.
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "plans",
        help="dry-run 실행 폴더의 상위 디렉터리",
    )
    # --execute일 때만 timestamp·raw register·SI 변환 측정값을 이곳에 저장한다.
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="실기체 실행별 측정 폴더의 상위 디렉터리",
    )
    args = parser.parse_args()
    config = load_from_args(args)
    require_execute_confirmation(args, config)
    # CLI에서 생략한 시험 조건만 YAML 기본값으로 채운다.
    amplitude = args.amplitude_rad or config.experiment.default_step_amplitude_rad
    hold = args.hold_sec or config.experiment.center_hold_sec
    joint = config.by_name[args.joint]
    # +A와 -A 중 하나라도 XML 관절 한계를 넘으면 Torque On 전에 즉시 차단한다.
    joint.validate_angle(joint.walking_rad + amplitude)
    joint.validate_angle(joint.walking_rad - amplitude)
    center = config.walking_pose()
    # 한 repeat는 각 hold_sec인 5단계이므로 표본 수는 hold×100×5×repeats다.
    test_samples = list(
        compact_joint_step(center, args.joint, amplitude, hold, args.repeats, 100)
    )
    stem = timestamped_stem(f"{args.joint}_step")
    common_metadata = {
            "valid_flag": False,
            "data_kind": "dry_run_plan" if not args.execute else "real_measurement",
            "joint": args.joint,
            "amplitude_rad": amplitude,
            "hold_sec": hold,
            "repeats": args.repeats,
            "command_rate_hz": config.bus.command_rate_hz,
            "state_read_rate_hz": config.bus.state_read_rate_hz,
            "hardware_error_read_rate_hz": (
                config.bus.hardware_error_read_rate_hz
            ),
            "recorded_transition": False,
            "expected_sample_count": len(test_samples),
            "config": str(config.source),
        }
    if not args.execute:
        # dry-run 한 번도 독립 폴더 하나에 plan.csv와 metadata.json으로 묶는다.
        run_dir = args.output_dir / stem
        plan_path = run_dir / "plan.csv"
        count = write_plan_csv(plan_path, test_samples)
        write_metadata(run_dir / "metadata.json", common_metadata)
        print("DRY-RUN: 포트 접근, Torque On, Goal Position 전송을 하지 않았습니다.")
        print(f"{args.joint} step 계획 {count} samples: {plan_path}")
        return

    # 실제 측정 한 번마다 새 폴더를 만들고 측정 CSV와 metadata를 함께 둔다.
    run_dir = args.raw_output_dir / stem
    raw_path = run_dir / "telemetry.csv"
    metadata_path = run_dir / "metadata.json"
    try:
        with DynamixelBus(config) as bus, RawCsvRecorder(raw_path, config) as recorder:
            print(f"Ping OK: {bus.ping_all()}")
            current_pose = states_to_pose(config, bus.read_state())
            # Torque On 순간 점프하지 않도록 먼저 현재 실측각을 Goal Position으로 쓴다.
            bus.write_goal_ticks(
                pose_to_ticks(config, current_pose, allow_outside_limits=True)
            )
            bus.set_torque(True)
            try:
                # 안전한 자세 전환은 실행하지만 식별 자료가 아니므로 recorder를 넘기지 않는다.
                transition = smooth_transition(
                    current_pose, center, config.experiment.transition_sec, 100
                )
                run_trajectory(
                    bus,
                    config,
                    transition,
                    allow_outside_limits=True,
                )
                # 전환 뒤에는 XML 한계를 강제한 단일관절 5단계 시험을 실행한다.
                run_trajectory(bus, config, iter(test_samples), recorder.write)
            finally:
                # 정상 종료와 예외 종료 모두 실기체 Torque를 반드시 끈다.
                bus.set_torque(False)
    except BaseException as exc:
        write_metadata(metadata_path, {**common_metadata, "invalid_reason": repr(exc)})
        raise
    write_metadata(
        metadata_path,
        {**common_metadata, "valid_flag": True, "invalid_reason": ""},
    )
    print("시험 종료 및 Torque Off 완료")
    print(f"실행 폴더: {run_dir}")
    print(f"100 Hz raw/SI telemetry: {raw_path}")


if __name__ == "__main__":
    main()
