from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


MUJOCO_DOF_ORDER = tuple(
    [f"RL{i}_joint" for i in range(1, 7)]
    + [f"LL{i}_joint" for i in range(1, 7)]
)
TICKS_PER_REV = 4096


@dataclass(frozen=True)
class BusConfig:
    port: str
    baudrate: int
    protocol_version: float
    command_rate_hz: int
    state_read_rate_hz: int
    hardware_error_read_rate_hz: int


@dataclass(frozen=True)
class ExperimentConfig:
    transition_sec: float
    center_hold_sec: float
    default_step_amplitude_rad: float


@dataclass(frozen=True)
class JointConfig:
    name: str
    motor_id: int | None
    zero_tick: int | None
    direction: int | None
    min_rad: float
    max_rad: float
    walking_rad: float

    @property
    def hardware_ready(self) -> bool:
        return (
            self.motor_id is not None
            and self.zero_tick is not None
            and self.direction in (-1, 1)
        )

    def validate_angle(self, angle_rad: float) -> None:
        if not self.min_rad <= angle_rad <= self.max_rad:
            raise ValueError(
                f"{self.name}: {angle_rad:+.6f} rad가 XML 범위 "
                f"[{self.min_rad:+.6f}, {self.max_rad:+.6f}] 밖입니다."
            )

    def rad_to_tick(
        self, angle_rad: float, *, allow_outside_limits: bool = False
    ) -> int:
        if not self.hardware_ready:
            raise RuntimeError(f"{self.name}: id/zero_tick/direction이 확정되지 않았습니다.")
        if not allow_outside_limits:
            self.validate_angle(angle_rad)
        assert self.zero_tick is not None and self.direction is not None
        tick = round(
            self.zero_tick
            + self.direction * angle_rad * TICKS_PER_REV / (2.0 * 3.141592653589793)
        )
        if not 0 <= tick <= 4095:
            raise ValueError(f"{self.name}: 변환된 Goal Position {tick} tick이 [0,4095] 밖입니다.")
        return tick

    def tick_to_rad(self, tick: int) -> float:
        if not self.hardware_ready:
            raise RuntimeError(f"{self.name}: id/zero_tick/direction이 확정되지 않았습니다.")
        assert self.zero_tick is not None and self.direction is not None
        return self.direction * (tick - self.zero_tick) * 2.0 * 3.141592653589793 / TICKS_PER_REV


@dataclass(frozen=True)
class RobotConfig:
    source: Path
    bus: BusConfig
    experiment: ExperimentConfig
    joints: tuple[JointConfig, ...]

    @property
    def by_name(self) -> dict[str, JointConfig]:
        return {joint.name: joint for joint in self.joints}

    @property
    def hardware_ready(self) -> bool:
        ids = [joint.motor_id for joint in self.joints]
        return all(joint.hardware_ready for joint in self.joints) and len(set(ids)) == len(ids)

    def require_hardware_ready(self) -> None:
        missing = [joint.name for joint in self.joints if not joint.hardware_ready]
        ids = [joint.motor_id for joint in self.joints if joint.motor_id is not None]
        duplicate_ids = sorted({motor_id for motor_id in ids if ids.count(motor_id) > 1})
        if missing or duplicate_ids:
            details = []
            if missing:
                details.append("미확정 관절=" + ", ".join(missing))
            if duplicate_ids:
                details.append("중복 ID=" + ", ".join(map(str, duplicate_ids)))
            raise RuntimeError(
                "실기체 명령을 차단했습니다: " + "; ".join(details)
                + ". configs/jandi_mx106.yaml을 실측값으로 채우세요."
            )

    def walking_pose(self) -> dict[str, float]:
        return {joint.name: joint.walking_rad for joint in self.joints}


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"필수 config 항목이 없습니다: {key}")
    return mapping[key]


def load_robot_config(path: str | Path) -> RobotConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text())
    bus_raw = _required(raw, "bus")
    exp_raw = _required(raw, "experiment")
    joints_raw = _required(raw, "joints")
    if tuple(joints_raw) != MUJOCO_DOF_ORDER:
        raise ValueError(
            "joints 순서는 MuJoCo 순서 RL1..RL6, LL1..LL6여야 합니다. "
            f"현재={tuple(joints_raw)}"
        )

    joints = []
    for name in MUJOCO_DOF_ORDER:
        item = joints_raw[name]
        joint = JointConfig(
            name=name,
            motor_id=item.get("id"),
            zero_tick=item.get("zero_tick"),
            direction=item.get("direction"),
            min_rad=float(_required(item, "min_rad")),
            max_rad=float(_required(item, "max_rad")),
            walking_rad=float(_required(item, "walking_rad")),
        )
        joint.validate_angle(joint.walking_rad)
        joints.append(joint)

    config = RobotConfig(
        source=source,
        bus=BusConfig(
            port=str(_required(bus_raw, "port")),
            baudrate=int(_required(bus_raw, "baudrate")),
            protocol_version=float(_required(bus_raw, "protocol_version")),
            command_rate_hz=int(_required(bus_raw, "command_rate_hz")),
            state_read_rate_hz=int(_required(bus_raw, "state_read_rate_hz")),
            hardware_error_read_rate_hz=int(
                _required(bus_raw, "hardware_error_read_rate_hz")
            ),
        ),
        experiment=ExperimentConfig(
            transition_sec=float(_required(exp_raw, "transition_sec")),
            center_hold_sec=float(_required(exp_raw, "center_hold_sec")),
            default_step_amplitude_rad=float(
                _required(exp_raw, "default_step_amplitude_rad")
            ),
        ),
        joints=tuple(joints),
    )
    if config.bus.command_rate_hz != 100:
        raise ValueError("현재 Real2Sim 명령 계약은 정확히 100 Hz입니다.")
    if (
        config.bus.state_read_rate_hz != 99
        or config.bus.hardware_error_read_rate_hz != 1
        or config.bus.state_read_rate_hz
        + config.bus.hardware_error_read_rate_hz
        != config.bus.command_rate_hz
    ):
        raise ValueError(
            "현재 Real2Sim 수신 계약은 매초 state 99회 + hardware error 1회입니다."
        )
    return config
