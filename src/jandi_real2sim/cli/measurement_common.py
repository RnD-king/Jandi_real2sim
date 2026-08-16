from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jandi_real2sim.config import MUJOCO_DOF_ORDER, RobotConfig
from jandi_real2sim.dynamixel_bus import DynamixelBus, MotorState
from jandi_real2sim.experiment import (
    HARDWARE_ERROR_SAMPLE,
    STATE_SAMPLE,
    acquisition_kind,
    pose_to_ticks,
    run_trajectory,
    states_to_pose,
)
from jandi_real2sim.records import (
    RawCsvRecorder,
    timestamped_stem,
    write_metadata,
    write_plan_csv,
)
from jandi_real2sim.trajectory import TrajectorySample, smooth_transition

from .common import PROJECT_ROOT, add_config_argument, require_execute_confirmation


def add_measurement_arguments(parser: argparse.ArgumentParser) -> None:
    add_config_argument(parser)
    parser.add_argument(
        "--pose-id",
        default="A",
        help="metadata에 기록할 자세 ID; 기본 A는 walking pose",
    )
    parser.add_argument(
        "--pose-json",
        type=Path,
        help="12관절 중심자세 JSON; 생략하면 config의 walking pose",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="실기체에서 실행; 생략 시 포트·Torque를 건드리지 않는 dry-run",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help="실기체 실행 시 정확히 MOVE_JANDI 입력",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "plans",
        help="dry-run 실행 폴더의 상위 디렉터리",
    )
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="실측 실행 폴더의 상위 디렉터리",
    )
    parser.add_argument(
        "--position-p-gain",
        type=int,
        help="Torque On 전 전 모터 Position P Gain RAM 값 설정 및 readback 검증",
    )
    parser.add_argument(
        "--position-i-gain",
        type=int,
        help="Torque On 전 전 모터 Position I Gain RAM 값 설정 및 readback 검증",
    )
    parser.add_argument(
        "--position-d-gain",
        type=int,
        help="Torque On 전 전 모터 Position D Gain RAM 값 설정 및 readback 검증",
    )
    parser.add_argument(
        "--position-pid-json",
        type=Path,
        help="관절별 {joint: {p, i, d}} JSON; 전역 gain 옵션과 함께 사용 불가",
    )
    parser.add_argument(
        "--campaign-experiment-name",
        help="자동 campaign 안에서 사용자가 지정한 실험 이름을 metadata에 기록",
    )
    parser.add_argument(
        "--safety-json",
        type=Path,
        help="99 Hz live limit과 Hardware Error 즉시 중단 조건 JSON",
    )


def load_pose(config: RobotConfig, pose_json: Path | None) -> dict[str, float]:
    if pose_json is None:
        return config.walking_pose()
    raw = json.loads(pose_json.expanduser().read_text())
    if not isinstance(raw, dict):
        raise ValueError("pose JSON은 {joint_name: angle_rad} 객체여야 합니다.")
    missing = [name for name in MUJOCO_DOF_ORDER if name not in raw]
    extra = [name for name in raw if name not in MUJOCO_DOF_ORDER]
    if missing or extra:
        raise ValueError(f"pose JSON 관절 불일치: missing={missing}, extra={extra}")
    pose = {name: float(raw[name]) for name in MUJOCO_DOF_ORDER}
    for joint in config.joints:
        joint.validate_angle(pose[joint.name])
    return pose


def _requested_pid_by_joint(
    args: argparse.Namespace, config: RobotConfig
) -> dict[str, dict[str, int]] | None:
    pid_json = getattr(args, "position_pid_json", None)
    global_values = {
        "p": getattr(args, "position_p_gain", None),
        "i": getattr(args, "position_i_gain", None),
        "d": getattr(args, "position_d_gain", None),
    }
    if pid_json is not None and any(value is not None for value in global_values.values()):
        raise ValueError("--position-pid-json과 전역 P/I/D 옵션은 함께 사용할 수 없습니다.")
    if pid_json is None:
        if not any(value is not None for value in global_values.values()):
            return None
        return {
            joint.name: {
                name: int(value)
                for name, value in global_values.items()
                if value is not None
            }
            for joint in config.joints
        }
    raw = json.loads(Path(pid_json).expanduser().read_text())
    if tuple(raw) != MUJOCO_DOF_ORDER:
        raise ValueError("position PID JSON 관절 순서는 RL1..RL6, LL1..LL6여야 합니다.")
    resolved: dict[str, dict[str, int]] = {}
    for joint in config.joints:
        values = raw[joint.name]
        if set(values) != {"p", "i", "d"}:
            raise ValueError(f"{joint.name}: PID JSON에는 p/i/d가 모두 필요합니다.")
        resolved[joint.name] = {name: int(value) for name, value in values.items()}
    return resolved


def validate_samples(config: RobotConfig, samples: Sequence[TrajectorySample]) -> None:
    if not samples:
        raise ValueError("시험 trajectory가 비어 있습니다.")
    for expected_cycle, sample in enumerate(samples):
        if sample.cycle_index != expected_cycle:
            raise ValueError(
                f"cycle_index가 연속적이지 않습니다: expected={expected_cycle}, "
                f"actual={sample.cycle_index}"
            )
        if tuple(sample.q_cmd_rad) != MUJOCO_DOF_ORDER:
            raise ValueError(f"cycle {sample.cycle_index}: 관절 순서 또는 구성이 다릅니다.")
        for joint in config.joints:
            joint.validate_angle(float(sample.q_cmd_rad[joint.name]))


def max_command_speed(
    samples: Sequence[TrajectorySample], rate_hz: int, joint_name: str
) -> float:
    return max(
        (
            abs(
                samples[index].q_cmd_rad[joint_name]
                - samples[index - 1].q_cmd_rad[joint_name]
            )
            * rate_hz
            for index in range(1, len(samples))
        ),
        default=0.0,
    )


def _state_snapshot(
    config: RobotConfig, states: Mapping[int, MotorState]
) -> tuple[dict[str, float], dict[str, float]]:
    velocity_unit = 0.229 * 2.0 * 3.141592653589793 / 60.0
    q_init: dict[str, float] = {}
    dq_init: dict[str, float] = {}
    for joint in config.joints:
        assert joint.motor_id is not None and joint.direction is not None
        state = states[joint.motor_id]
        q_init[joint.name] = joint.tick_to_rad(state.present_position_tick)
        dq_init[joint.name] = (
            joint.direction * state.present_velocity_raw * velocity_unit
        )
    return q_init, dq_init


@dataclass
class LiveSafetyMonitor:
    config: RobotConfig
    limits: dict[str, float | int]
    counters: dict[tuple[int, str], int] = field(default_factory=dict)

    def check(
        self,
        sample: TrajectorySample,
        states: Mapping[int, MotorState] | None,
        hardware_errors: Mapping[int, int] | None,
    ) -> None:
        if hardware_errors is not None:
            active = {
                motor_id: value
                for motor_id, value in hardware_errors.items()
                if value != 0
            }
            if active:
                raise RuntimeError(f"LIVE SAFETY Hardware Error: {active}")
        if states is None:
            return
        required = int(self.limits["consecutive_state_samples"])
        checks: list[tuple[int, str, float, float, bool]] = []
        for joint in self.config.joints:
            assert joint.motor_id is not None
            state = states[joint.motor_id]
            values = (
                (
                    "temperature_c",
                    float(state.temperature_c),
                    float(self.limits["max_temperature_c"]),
                    state.temperature_c >= float(self.limits["max_temperature_c"]),
                ),
                (
                    "input_voltage_v",
                    state.input_voltage_raw * 0.1,
                    float(self.limits["min_input_voltage_v"]),
                    state.input_voltage_raw * 0.1 <= float(self.limits["min_input_voltage_v"]),
                ),
                (
                    "abs_current_a",
                    abs(state.present_current_raw * 0.00336),
                    float(self.limits["max_abs_current_a"]),
                    abs(state.present_current_raw * 0.00336) >= float(self.limits["max_abs_current_a"]),
                ),
                (
                    "abs_pwm_percent",
                    abs(state.present_pwm_raw * 0.113),
                    float(self.limits["max_abs_pwm_percent"]),
                    abs(state.present_pwm_raw * 0.113) >= float(self.limits["max_abs_pwm_percent"]),
                ),
                (
                    "abs_position_error_rad",
                    abs(
                        sample.q_cmd_rad[joint.name]
                        - joint.tick_to_rad(state.present_position_tick)
                    ),
                    float(self.limits["max_abs_position_error_rad"]),
                    abs(
                        sample.q_cmd_rad[joint.name]
                        - joint.tick_to_rad(state.present_position_tick)
                    ) >= float(self.limits["max_abs_position_error_rad"]),
                ),
            )
            checks.extend(
                (joint.motor_id, name, actual, limit, violated)
                for name, actual, limit, violated in values
            )
        for motor_id, name, actual, limit, violated in checks:
            key = (motor_id, name)
            self.counters[key] = self.counters.get(key, 0) + 1 if violated else 0
            if self.counters[key] >= required:
                comparison = "min" if name == "input_voltage_v" else "max"
                raise RuntimeError(
                    "LIVE SAFETY limit: "
                    f"ID={motor_id} metric={name} actual={actual:.6f} "
                    f"{comparison}_limit={limit:.6f} consecutive={self.counters[key]}"
                )


def _load_safety(args: argparse.Namespace) -> dict[str, float | int] | None:
    path = getattr(args, "safety_json", None)
    if path is None:
        return None
    raw = json.loads(Path(path).expanduser().read_text())
    required = {
        "max_temperature_c",
        "min_input_voltage_v",
        "max_abs_current_a",
        "max_abs_pwm_percent",
        "max_abs_position_error_rad",
        "consecutive_state_samples",
    }
    if set(raw) != required:
        raise ValueError(f"safety JSON 항목 불일치: expected={required}, actual={set(raw)}")
    if int(raw["consecutive_state_samples"]) < 1:
        raise ValueError("consecutive_state_samples는 1 이상이어야 합니다.")
    for name in (
        "max_temperature_c",
        "min_input_voltage_v",
        "max_abs_current_a",
        "max_abs_pwm_percent",
        "max_abs_position_error_rad",
    ):
        if float(raw[name]) <= 0:
            raise ValueError(f"{name}은 0보다 커야 합니다.")
    return raw


def collect_or_plan(
    args: argparse.Namespace,
    config: RobotConfig,
    *,
    experiment_type: str,
    samples: Sequence[TrajectorySample],
    center_pose: Mapping[str, float],
    metadata: dict[str, Any],
    name_suffix: str,
) -> Path:
    """동일 trajectory를 dry-run 계획 또는 안전한 실측 run으로 저장한다."""
    require_execute_confirmation(args, config)
    validate_samples(config, samples)
    kinds = Counter(acquisition_kind(sample.cycle_index, config) for sample in samples)
    stem = timestamped_stem(
        f"{experiment_type}_{args.pose_id}_{name_suffix}"
    )
    requested_pid = _requested_pid_by_joint(args, config)
    safety_limits = _load_safety(args)
    common_metadata: dict[str, Any] = {
        "valid_flag": False,
        "invalid_reason": "",
        "data_kind": "real_measurement" if args.execute else "dry_run_plan",
        "experiment_type": experiment_type,
        "pose_id": args.pose_id,
        "pose_rad": dict(center_pose),
        "command_rate_hz": config.bus.command_rate_hz,
        "state_read_rate_hz": config.bus.state_read_rate_hz,
        "hardware_error_read_rate_hz": config.bus.hardware_error_read_rate_hz,
        "recorded_transition": False,
        "expected_sample_count": len(samples),
        "expected_state_samples": kinds[STATE_SAMPLE],
        "expected_hardware_error_samples": kinds[HARDWARE_ERROR_SAMPLE],
        "config": str(config.source),
        "campaign_experiment_name": getattr(args, "campaign_experiment_name", None),
        "requested_position_pid": {
            "p": getattr(args, "position_p_gain", None),
            "i": getattr(args, "position_i_gain", None),
            "d": getattr(args, "position_d_gain", None),
        },
        "requested_position_pid_by_joint": requested_pid,
        "live_safety_limits": safety_limits,
        **metadata,
    }

    if not args.execute:
        run_dir = args.output_dir / stem
        plan_path = run_dir / "plan.csv"
        count = write_plan_csv(plan_path, samples)
        write_metadata(
            run_dir / "metadata.json",
            {**common_metadata, "expected_sample_count": count},
        )
        print("DRY-RUN: 포트 접근, Torque On, Goal Position 전송을 하지 않았습니다.")
        print(f"실험 계획 {count} samples: {plan_path}")
        return run_dir

    run_dir = args.raw_output_dir / stem
    raw_path = run_dir / "telemetry.csv"
    metadata_path = run_dir / "metadata.json"
    runtime_metadata = dict(common_metadata)
    try:
        with DynamixelBus(config) as bus, RawCsvRecorder(raw_path, config) as recorder:
            runtime_metadata["ping_models"] = bus.ping_all()
            print(f"Ping OK: {runtime_metadata['ping_models']}")
            # RAM gain은 전원 재인가 시 초기화되므로 매 run Torque Off에서 다시 쓴다.
            bus.set_torque(False)
            if requested_pid is None:
                requested_settings = {}
            else:
                requested_settings = bus.write_position_pid_gains_by_motor(
                    {
                        int(joint.motor_id): {
                            f"position_{name}_gain": value
                            for name, value in requested_pid[joint.name].items()
                        }
                        for joint in config.joints
                        if joint.motor_id is not None
                    }
                )
            # 실제 MX 내부 설정이 달라지면 식별 대상 시스템도 달라지므로 run마다 저장한다.
            runtime_metadata["actuator_settings"] = (
                requested_settings or bus.read_actuator_settings()
            )
            current_states = bus.read_state()
            current_pose = states_to_pose(config, current_states)
            bus.write_goal_ticks(
                pose_to_ticks(config, current_pose, allow_outside_limits=True)
            )
            bus.set_torque(True)
            try:
                monitor = (
                    LiveSafetyMonitor(config, safety_limits)
                    if safety_limits is not None
                    else None
                )

                def check_only(
                    sample: TrajectorySample,
                    _tx_time_ns: int,
                    _rx_time_ns: int,
                    _acquisition_kind: str,
                    states: Mapping[int, MotorState] | None,
                    hardware_errors: Mapping[int, int] | None,
                    _overrun_ns: int,
                ) -> None:
                    if monitor is not None:
                        monitor.check(sample, states, hardware_errors)

                transition = smooth_transition(
                    current_pose,
                    center_pose,
                    config.experiment.transition_sec,
                    config.bus.command_rate_hz,
                )
                # 전환은 안전과 자세 재현을 위해 수행하지만 식별 CSV에는 넣지 않는다.
                run_trajectory(
                    bus,
                    config,
                    transition,
                    check_only,
                    allow_outside_limits=True,
                )
                initial_states = bus.read_state()
                q_init, dq_init = _state_snapshot(config, initial_states)
                runtime_metadata["q_init_rad"] = q_init
                runtime_metadata["dq_init_rad_s"] = dq_init
                def record_and_check(
                    sample: TrajectorySample,
                    tx_time_ns: int,
                    rx_time_ns: int,
                    acquisition_kind: str,
                    states: Mapping[int, MotorState] | None,
                    hardware_errors: Mapping[int, int] | None,
                    overrun_ns: int,
                ) -> None:
                    # 한계에 걸린 마지막 표본도 원인 분석을 위해 먼저 기록한다.
                    recorder.write(
                        sample,
                        tx_time_ns,
                        rx_time_ns,
                        acquisition_kind,
                        states,
                        hardware_errors,
                        overrun_ns,
                    )
                    if monitor is not None:
                        monitor.check(sample, states, hardware_errors)

                run_trajectory(bus, config, iter(samples), record_and_check)
            finally:
                bus.set_torque(False)
    except BaseException as exc:
        write_metadata(
            metadata_path,
            {**runtime_metadata, "valid_flag": False, "invalid_reason": repr(exc)},
        )
        raise
    write_metadata(
        metadata_path,
        {**runtime_metadata, "valid_flag": True, "invalid_reason": ""},
    )
    print("시험 종료 및 Torque Off 완료")
    print(f"실행 폴더: {run_dir}")
    print(f"telemetry: {raw_path}")
    return run_dir
