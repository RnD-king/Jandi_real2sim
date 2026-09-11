"""Readback and low-amplitude sign calibration for the Mode-3 BAM bench."""

from __future__ import annotations

import argparse
from datetime import datetime
import glob
import json
import math
from pathlib import Path
import statistics
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import yaml

from .bus import Mode3Bus
from .config import DEFAULT_CAMPAIGN


BAUDRATE_CANDIDATES = (4_500_000, 4_000_000, 3_000_000, 2_000_000, 1_000_000, 115_200, 57_600, 9_600)
SIGN_CONFIRMATION = "MOVE_MX106_MODE3_CALIBRATION"
TICK_RAD = 2.0 * math.pi / 4096.0
CURRENT_A_PER_RAW = 0.00336
PWM_FRACTION_PER_RAW = 0.00113
VELOCITY_RAD_S_PER_RAW = 0.229 * 2.0 * math.pi / 60.0


READBACK_NAMES = (
    "firmware_version", "return_delay_time_raw", "homing_offset_raw", "drive_mode",
    "operating_mode", "temperature_limit_c", "max_voltage_limit_raw",
    "min_voltage_limit_raw", "pwm_limit_raw", "current_limit_raw", "velocity_limit_raw",
    "max_position_limit_raw", "min_position_limit_raw", "position_d_gain",
    "position_i_gain", "position_p_gain", "feedforward_2nd_gain",
    "feedforward_1st_gain", "profile_acceleration", "profile_velocity",
    "status_return_level_raw", "bus_watchdog_raw", "hardware_error", "torque_enable",
)


def candidate_ports() -> tuple[str, ...]:
    paths = glob.glob("/dev/serial/by-id/*") + glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    chosen: dict[str, str] = {}
    for name in paths:
        resolved = str(Path(name).resolve())
        # Prefer the stable /dev/serial/by-id spelling over ttyUSB/ttyACM.
        if resolved not in chosen or "/by-id/" in name:
            chosen[resolved] = name
    return tuple(chosen.values())


def _scan(port_name: str, baudrates: tuple[int, ...], requested_id: int | None) -> list[dict[str, int | str]]:
    import dynamixel_sdk as sdk

    found: list[dict[str, int | str]] = []
    port = sdk.PortHandler(port_name)
    packet = sdk.PacketHandler(2.0)
    if not port.openPort():
        return found
    try:
        for baudrate in baudrates:
            if not port.setBaudRate(baudrate):
                continue
            devices, result = packet.broadcastPing(port)
            if result not in (sdk.COMM_SUCCESS, sdk.COMM_RX_TIMEOUT):
                continue
            for motor_id, values in devices.items():
                if requested_id is None or int(motor_id) == requested_id:
                    found.append({"serial_device": port_name, "baudrate": baudrate,
                                  "motor_id": int(motor_id), "model_number": int(values[0]),
                                  "firmware_version": int(values[1])})
    finally:
        port.closePort()
    return found


def discover(port: str | None, baudrate: int | None, motor_id: int | None) -> dict[str, int | str]:
    ports = (port,) if port else candidate_ports()
    if not ports:
        raise RuntimeError("DYNAMIXEL serial port 후보가 없습니다. --port로 지정하십시오.")
    baudrates = (baudrate,) if baudrate else BAUDRATE_CANDIDATES
    found = [item for name in ports for item in _scan(name, baudrates, motor_id)]
    unique = {(str(item["serial_device"]), int(item["baudrate"]), int(item["motor_id"])): item for item in found}
    if not unique:
        raise RuntimeError("Protocol 2.0 DYNAMIXEL을 찾지 못했습니다.")
    if len(unique) != 1:
        summary = ", ".join(f"{p}@{b}:ID{i}" for p, b, i in unique)
        raise RuntimeError(f"모터가 하나로 결정되지 않습니다: {summary}. --port/--baudrate/--motor-id를 지정하십시오.")
    return next(iter(unique.values()))


def _cfg_for(device: dict[str, int | str]) -> Any:
    return SimpleNamespace(hardware={"serial_device": device["serial_device"],
                                     "protocol_version": 2.0,
                                     "baudrate": device["baudrate"],
                                     "motor_id": device["motor_id"]})


def _dominant_sign(values: list[int], *, minimum: int) -> int:
    useful = [value for value in values if abs(value) >= minimum]
    if not useful:
        raise RuntimeError("전류/PWM 부호를 판별할 신호가 너무 작습니다. 혼을 장착하고 다시 시도하십시오.")
    median = statistics.median(useful)
    return 1 if median > 0 else -1


def infer_signs(raw_positive_is_joint_positive: bool, currents: list[int], pwms: list[int]) -> dict[str, int]:
    direction = 1 if raw_positive_is_joint_positive else -1
    return {
        "direction": direction,
        "current_direction": direction * _dominant_sign(currents, minimum=2),
        "pwm_direction": direction * _dominant_sign(pwms, minimum=2),
        # q=0 is the upright position: tau_g = +mgl*sin(q).
        "gravity_torque_sign": 1,
    }


def verify_sign_test_configuration(bus: Mode3Bus) -> None:
    """Reject motion unless the motor is in the canonical experiment mode."""

    if bus.read("operating_mode") != 3:
        raise RuntimeError("Operating Mode가 3이 아닙니다. Wizard에서 Mode 3으로 설정 후 재실행하십시오.")
    if bus.read("position_p_gain") != 850:
        raise RuntimeError("Position P Gain이 850이 아닙니다. 850으로 설정 후 재실행하십시오.")


def capture_upright_zero(bus: Mode3Bus, sample_count: int = 20) -> int:
    """Capture the current raw position without enabling torque or commanding motion."""

    if sample_count < 3:
        raise ValueError("zero sample count는 3 이상이어야 합니다.")
    bus.torque(False)
    return round(statistics.median(
        bus.read_state().present_position_raw for _ in range(sample_count)
    ))


def jog_raw_positive(bus: Mode3Bus, zero_raw: int, jog_ticks: int) -> tuple[list[int], list[int]]:
    """Make one bounded raw-positive jog and always finish with torque disabled."""

    if jog_ticks <= 0 or jog_ticks > 64:
        raise ValueError("jog-ticks는 1..64 범위여야 합니다.")
    verify_sign_test_configuration(bus)
    minimum = bus.read("min_position_limit_raw")
    maximum = bus.read("max_position_limit_raw")
    if not minimum <= zero_raw <= zero_raw + jog_ticks <= maximum:
        raise RuntimeError(
            f"방향 시험 Goal이 하드웨어 위치 한계를 벗어납니다: "
            f"[{zero_raw}, {zero_raw + jog_ticks}] not in [{minimum}, {maximum}]"
        )
    bus.torque(False)
    bus.write("goal_position_raw", zero_raw)
    currents: list[int] = []
    pwms: list[int] = []
    try:
        bus.torque(True)
        for offset in range(1, jog_ticks + 1):
            bus.write("goal_position_raw", zero_raw + offset)
            state = bus.read_state()
            currents.append(state.present_current_raw)
            pwms.append(state.present_pwm_raw)
            hardware_error = bus.read_hardware_error()
            if hardware_error:
                raise RuntimeError(f"Hardware Error Status={hardware_error}")
            time.sleep(0.01)
        time.sleep(0.25)
        for offset in range(jog_ticks - 1, -1, -1):
            bus.write("goal_position_raw", zero_raw + offset)
            time.sleep(0.01)
    finally:
        bus.torque(False)
    return currents, pwms


def sign_test(bus: Mode3Bus, jog_ticks: int) -> tuple[int, dict[str, int]]:
    verify_sign_test_configuration(bus)
    input("추가 수직 위를 보는 정확한 q=0 위치에 놓고 ENTER를 누르십시오: ")
    zero_raw = capture_upright_zero(bus)
    currents, pwms = jog_raw_positive(bus, zero_raw, jog_ticks)
    answer = input("방금 raw tick 증가 방향이 정의한 +관절 방향이었습니까? [y/n]: ").strip().lower()
    if answer not in ("y", "yes", "n", "no"):
        raise RuntimeError("y 또는 n으로 답해야 합니다. 결과를 적용하지 않습니다.")
    return zero_raw, infer_signs(answer in ("y", "yes"), currents, pwms)


def readback_device(device: dict[str, int | str]) -> tuple[dict[str, int], dict[str, int | float]]:
    """Torque-off readback shared by the terminal and desktop interfaces."""

    with Mode3Bus(_cfg_for(device)) as bus:
        bus.torque(False)
        model = bus.ping()
        if model != int(device["model_number"]):
            raise RuntimeError("Ping model number changed during calibration")
        snapshot = {name: bus.read(name) for name in READBACK_NAMES}
        state = bus.read_state()
    present = {
        "position_raw": state.present_position_raw,
        "input_voltage_v": state.input_voltage_raw * 0.1,
        "temperature_c": state.temperature_c,
    }
    return snapshot, present


def calibration_report(device: dict[str, int | str], snapshot: dict[str, int],
                       present: dict[str, int | float], zero_raw: int | None,
                       signs: dict[str, int] | None) -> dict[str, Any]:
    return {
        "created_at": datetime.now().astimezone().isoformat(), "device": device,
        "register_readback": snapshot, "present": present,
        "engineering_hardware_limits": engineering_limits(snapshot),
        "captured_encoder_zero_raw": zero_raw, "inferred_signs": signs,
        "not_auto_selected": [
            "bus_watchdog_raw", "software safety thresholds", "experiment angle range",
            "load masses/distances", "arm mass/COM/inertia",
        ],
    }


def save_calibration_report(config: Path, report: dict[str, Any]) -> Path:
    report_root = config.resolve().parents[2] / "data/calibration"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_mode3_calibration.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report_path


def engineering_limits(snapshot: dict[str, int]) -> dict[str, float]:
    return {
        "hardware_temperature_limit_c": float(snapshot["temperature_limit_c"]),
        "hardware_min_voltage_v": snapshot["min_voltage_limit_raw"] * 0.1,
        "hardware_max_voltage_v": snapshot["max_voltage_limit_raw"] * 0.1,
        "hardware_current_limit_A": snapshot["current_limit_raw"] * CURRENT_A_PER_RAW,
        "hardware_pwm_limit_fraction": snapshot["pwm_limit_raw"] * PWM_FRACTION_PER_RAW,
        "hardware_velocity_limit_rad_s": snapshot["velocity_limit_raw"] * VELOCITY_RAD_S_PER_RAW,
    }


def _write_yaml(path: Path, section: str, values: dict[str, Any]) -> None:
    document = yaml.safe_load(path.read_text())
    document[section].update(values)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".candidate.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with open(fd, "w") as stream:
            yaml.safe_dump(document, stream, sort_keys=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def apply_readback(config: Path, device: dict[str, int | str], snapshot: dict[str, int],
                   zero_raw: int | None, signs: dict[str, int] | None) -> None:
    root = config.parent
    hardware = {
        "serial_device": str(device["serial_device"]), "protocol_version": 2.0,
        "baudrate": int(device["baudrate"]), "motor_id": int(device["motor_id"]),
        "expected_model_number": int(device["model_number"]),
        "expected_homing_offset_raw": int(snapshot["homing_offset_raw"]),
    }
    if zero_raw is not None:
        hardware["encoder_zero_raw"] = zero_raw
    if signs is not None:
        hardware.update({name: signs[name] for name in ("direction", "current_direction", "pwm_direction")})
    _write_yaml(root / "hardware.yaml", "hardware", hardware)
    _write_yaml(root / "controller.yaml", "mode3_registers", {
        "drive_mode": int(snapshot["drive_mode"]), "position_d_gain": int(snapshot["position_d_gain"]),
        "expected_pwm_limit_raw": int(snapshot["pwm_limit_raw"]),
        "expected_current_limit_raw": int(snapshot["current_limit_raw"]),
    })
    if signs is not None:
        _write_yaml(root / "bench.yaml", "bench", {"gravity_torque_sign": signs["gravity_torque_sign"]})


def main() -> None:
    parser = argparse.ArgumentParser(description="MX-106 Mode-3 readback and sign calibration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--port"); parser.add_argument("--baudrate", type=int); parser.add_argument("--motor-id", type=int)
    parser.add_argument("--sign-test", action="store_true", help="capture upright zero and perform a small raw-positive jog")
    parser.add_argument("--jog-ticks", type=int, default=32, help="1..64; default 32 ticks = 2.81 degrees")
    parser.add_argument("--execute", action="store_true"); parser.add_argument("--confirm")
    parser.add_argument("--apply", action="store_true", help="write measured values to hardware/controller YAML")
    args = parser.parse_args()
    if args.sign_test and (not args.execute or args.confirm != SIGN_CONFIRMATION):
        parser.error(f"--sign-test requires --execute --confirm {SIGN_CONFIRMATION}")

    device = discover(args.port, args.baudrate, args.motor_id)
    zero_raw = None; signs = None
    snapshot, present = readback_device(device)
    if args.sign_test:
        with Mode3Bus(_cfg_for(device)) as bus:
            zero_raw, signs = sign_test(bus, args.jog_ticks)
    report = calibration_report(device, snapshot, present, zero_raw, signs)
    report_path = save_calibration_report(args.config, report)
    if args.apply:
        apply_readback(args.config.resolve(), device, snapshot, zero_raw, signs)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report: {report_path}")
    print("YAML applied" if args.apply else "read-only report; YAML unchanged (use --apply to write measured values)")


if __name__ == "__main__":
    main()
