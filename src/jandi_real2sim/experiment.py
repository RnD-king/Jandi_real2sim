from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping

from .config import RobotConfig
from .dynamixel_bus import DynamixelBus, MotorState
from .trajectory import TrajectorySample


STATE_SAMPLE = "state"
HARDWARE_ERROR_SAMPLE = "hardware_error"


def acquisition_kind(cycle_index: int, config: RobotConfig) -> str:
    """각 1초의 마지막 슬롯만 Hardware Error에 할당한다."""
    slot = cycle_index % config.bus.command_rate_hz
    if slot >= config.bus.state_read_rate_hz:
        return HARDWARE_ERROR_SAMPLE
    return STATE_SAMPLE


def pose_to_ticks(
    config: RobotConfig,
    pose: Mapping[str, float],
    *,
    allow_outside_limits: bool = False,
) -> dict[int, int]:
    goals = {}
    for joint in config.joints:
        if joint.name not in pose:
            raise ValueError(f"관절 목표 누락: {joint.name}")
        assert joint.motor_id is not None
        goals[joint.motor_id] = joint.rad_to_tick(
            float(pose[joint.name]),
            allow_outside_limits=allow_outside_limits,
        )
    return goals


def states_to_pose(config: RobotConfig, states: Mapping[int, MotorState]) -> dict[str, float]:
    pose = {}
    for joint in config.joints:
        assert joint.motor_id is not None
        pose[joint.name] = joint.tick_to_rad(states[joint.motor_id].present_position_tick)
    return pose


def run_trajectory(
    bus: DynamixelBus,
    config: RobotConfig,
    samples: Iterable[TrajectorySample],
    on_sample: Callable[
        [
            TrajectorySample,
            int,
            int,
            str,
            Mapping[int, MotorState] | None,
            Mapping[int, int] | None,
            int,
        ],
        None,
    ] | None = None,
    *,
    allow_outside_limits: bool = False,
) -> int:
    """100 Hz 명령 후 매초 state 99회, Hardware Error 1회를 배타적으로 읽는다."""
    period = 1.0 / config.bus.command_rate_hz
    start_ns = time.monotonic_ns()
    count = 0
    for count, sample in enumerate(samples, start=1):
        deadline_ns = start_ns + sample.cycle_index * int(period * 1e9)
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1e9)
        tx_time_ns = time.monotonic_ns()
        bus.write_goal_ticks(
            pose_to_ticks(
                config,
                sample.q_cmd_rad,
                allow_outside_limits=allow_outside_limits,
            )
        )
        kind = acquisition_kind(sample.cycle_index, config)
        states: Mapping[int, MotorState] | None = None
        hardware_errors: Mapping[int, int] | None = None
        if kind == STATE_SAMPLE:
            states = bus.read_state()
        else:
            hardware_errors = bus.read_hardware_errors()
        rx_time_ns = time.monotonic_ns()
        overrun_ns = max(0, rx_time_ns - (deadline_ns + int(period * 1e9)))
        if on_sample is not None:
            on_sample(
                sample,
                tx_time_ns,
                rx_time_ns,
                kind,
                states,
                hardware_errors,
                overrun_ns,
            )
        errors = {
            motor_id: error
            for motor_id, error in (hardware_errors or {}).items()
            if error
        }
        if errors:
            raise RuntimeError(f"Hardware Error Status 감지: {errors}")
    return count
