from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO

from .config import MUJOCO_DOF_ORDER, RobotConfig
from .dynamixel_bus import MotorState
from .trajectory import TrajectorySample


def timestamped_stem(prefix: str) -> str:
    return datetime.now().strftime(f"%Y%m%d_%H%M%S_{prefix}")


def write_plan_csv(path: Path, samples: Iterable[TrajectorySample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["cycle_index", "time_s", "phase"] + [
        f"q_cmd_{name}_rad" for name in MUJOCO_DOF_ORDER
    ]
    count = 0
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row: dict[str, Any] = {
                "cycle_index": sample.cycle_index,
                "time_s": f"{sample.time_s:.9f}",
                "phase": sample.phase,
            }
            row.update(
                {f"q_cmd_{name}_rad": f"{sample.q_cmd_rad[name]:.9f}" for name in MUJOCO_DOF_ORDER}
            )
            writer.writerow(row)
            count += 1
    return count


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


class RawCsvRecorder:
    """100 Hz 실측 raw 값과 SI 변환값을 같은 행에 저장한다."""

    def __init__(self, path: Path, config: RobotConfig):
        self.path = path
        self.config = config
        self._stream: TextIO | None = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self) -> "RawCsvRecorder":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", newline="")
        fields = [
            "host_time_ns", "tx_time_ns", "rx_time_ns", "cycle_index",
            "time_s", "phase", "acquisition_kind", "overrun_ns",
        ]
        per_joint = (
            "q_cmd_sent_rad", "goal_position_tick", "q_present_rad",
            "present_position_tick", "dq_present_rad_s", "present_velocity_raw",
            "position_trajectory_rad", "position_trajectory_tick",
            "velocity_trajectory_rad_s", "velocity_trajectory_raw",
            "pwm_percent", "present_pwm_raw", "current_A", "present_current_raw",
            "input_voltage_V", "temperature_C", "realtime_tick_ms", "moving",
            "moving_status", "hardware_error",
        )
        for name in MUJOCO_DOF_ORDER:
            fields.extend(f"{name}/{field}" for field in per_joint)
        self._writer = csv.DictWriter(self._stream, fieldnames=fields)
        self._writer.writeheader()
        return self

    def __exit__(self, *_: object) -> None:
        if self._stream is not None:
            self._stream.close()
        self._stream = None
        self._writer = None

    def write(
        self,
        sample: TrajectorySample,
        tx_time_ns: int,
        rx_time_ns: int,
        acquisition_kind: str,
        states: Mapping[int, MotorState] | None,
        hardware_errors: Mapping[int, int] | None,
        overrun_ns: int,
    ) -> None:
        if self._writer is None:
            raise RuntimeError("RawCsvRecorder context가 열리지 않았습니다.")
        row: dict[str, Any] = {
            "host_time_ns": rx_time_ns,
            "tx_time_ns": tx_time_ns,
            "rx_time_ns": rx_time_ns,
            "cycle_index": sample.cycle_index,
            "time_s": f"{sample.time_s:.9f}",
            "phase": sample.phase,
            "acquisition_kind": acquisition_kind,
            "overrun_ns": overrun_ns,
        }
        velocity_unit = 0.229 * 2.0 * 3.141592653589793 / 60.0
        for joint in self.config.joints:
            assert joint.motor_id is not None and joint.direction is not None
            prefix = joint.name + "/"
            row.update(
                {
                    prefix + "q_cmd_sent_rad": f"{sample.q_cmd_rad[joint.name]:.9f}",
                    prefix + "goal_position_tick": joint.rad_to_tick(
                        sample.q_cmd_rad[joint.name], allow_outside_limits=True
                    ),
                    prefix + "hardware_error": (
                        hardware_errors[joint.motor_id]
                        if hardware_errors is not None
                        else ""
                    ),
                }
            )
            if states is not None:
                state = states[joint.motor_id]
                row.update(
                    {
                        prefix + "q_present_rad": f"{joint.tick_to_rad(state.present_position_tick):.9f}",
                        prefix + "present_position_tick": state.present_position_tick,
                        prefix + "dq_present_rad_s": f"{joint.direction * state.present_velocity_raw * velocity_unit:.9f}",
                        prefix + "present_velocity_raw": state.present_velocity_raw,
                        prefix + "position_trajectory_rad": f"{joint.tick_to_rad(state.position_trajectory_tick):.9f}",
                        prefix + "position_trajectory_tick": state.position_trajectory_tick,
                        prefix + "velocity_trajectory_rad_s": f"{joint.direction * state.velocity_trajectory_raw * velocity_unit:.9f}",
                        prefix + "velocity_trajectory_raw": state.velocity_trajectory_raw,
                        prefix + "pwm_percent": f"{state.present_pwm_raw * 0.113:.6f}",
                        prefix + "present_pwm_raw": state.present_pwm_raw,
                        prefix + "current_A": f"{state.present_current_raw * 0.00336:.9f}",
                        prefix + "present_current_raw": state.present_current_raw,
                        prefix + "input_voltage_V": f"{state.input_voltage_raw * 0.1:.3f}",
                        prefix + "temperature_C": state.temperature_c,
                        prefix + "realtime_tick_ms": state.realtime_tick_ms,
                        prefix + "moving": state.moving,
                        prefix + "moving_status": state.moving_status,
                    }
                )
        self._writer.writerow(row)
