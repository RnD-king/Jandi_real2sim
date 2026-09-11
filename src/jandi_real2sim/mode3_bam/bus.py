"""Self-contained Protocol-2.0 MX-106R I/O for the Mode-3 BAM bench."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any


ADDR = {
    "firmware_version": (6, 1, False), "return_delay_time_raw": (9, 1, False),
    "drive_mode": (10, 1, False), "operating_mode": (11, 1, False),
    "homing_offset_raw": (20, 4, True), "temperature_limit_c": (31, 1, False),
    "max_voltage_limit_raw": (32, 2, False), "min_voltage_limit_raw": (34, 2, False),
    "pwm_limit_raw": (36, 2, False),
    "current_limit_raw": (38, 2, False), "torque_enable": (64, 1, False),
    "velocity_limit_raw": (44, 4, False), "max_position_limit_raw": (48, 4, False),
    "min_position_limit_raw": (52, 4, False),
    "status_return_level_raw": (68, 1, False), "hardware_error": (70, 1, False),
    "position_d_gain": (80, 2, False), "position_i_gain": (82, 2, False),
    "position_p_gain": (84, 2, False), "feedforward_2nd_gain": (88, 2, False),
    "feedforward_1st_gain": (90, 2, False), "bus_watchdog_raw": (98, 1, True),
    "profile_acceleration": (108, 4, False), "profile_velocity": (112, 4, False),
    "goal_position_raw": (116, 4, True),
}


def _signed(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


@dataclass(frozen=True)
class State:
    goal_position_raw: int
    realtime_tick_raw: int
    moving: int
    moving_status: int
    present_pwm_raw: int
    present_current_raw: int
    present_velocity_raw: int
    present_position_raw: int
    velocity_trajectory_raw: int
    position_trajectory_raw: int
    input_voltage_raw: int
    temperature_c: int


@dataclass(frozen=True)
class GoalWrite:
    raw: int
    tx_before_ns: int
    tx_after_ns: int


@dataclass(frozen=True)
class TimedState:
    state: State
    read_before_ns: int
    read_after_ns: int


class Mode3Bus:
    """Block I/O and strict readback for one MX-106R in Position Mode (3)."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._sdk: Any | None = None
        self._port: Any | None = None
        self._packet: Any | None = None
        self._reader: Any | None = None
        self._goal_writer: Any | None = None

    def __enter__(self) -> "Mode3Bus":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> None:
        import dynamixel_sdk as sdk

        self._sdk = sdk
        self._port = sdk.PortHandler(str(self.cfg.hardware["serial_device"]))
        self._packet = sdk.PacketHandler(float(self.cfg.hardware["protocol_version"]))
        if not self._port.openPort():
            raise RuntimeError(f"포트를 열 수 없습니다: {self.cfg.hardware['serial_device']}")
        if not self._port.setBaudRate(int(self.cfg.hardware["baudrate"])):
            self.close()
            raise RuntimeError(f"baudrate 설정 실패: {self.cfg.hardware['baudrate']}")
        self._reader = sdk.GroupSyncRead(self._port, self._packet, 116, 31)
        if not self._reader.addParam(int(self.cfg.hardware["motor_id"])):
            self.close()
            raise RuntimeError("state GroupSyncRead addParam 실패")
        self._goal_writer = sdk.GroupSyncWrite(self._port, self._packet, 116, 4)

    def close(self) -> None:
        if self._port is not None:
            self._port.closePort()
        self._port = self._reader = self._goal_writer = None

    def _require(self) -> None:
        if self._port is None or self._packet is None or self._sdk is None:
            raise RuntimeError("DYNAMIXEL 포트가 열리지 않았습니다.")

    def ping(self) -> int:
        self._require()
        model, result, error = self._packet.ping(self._port, int(self.cfg.hardware["motor_id"]))
        if result != self._sdk.COMM_SUCCESS or error:
            raise RuntimeError(f"Ping 실패: comm={result}, error={error}")
        return int(model)

    def read(self, name: str) -> int:
        self._require()
        address, length, signed = ADDR[name]
        method = {1: self._packet.read1ByteTxRx, 2: self._packet.read2ByteTxRx,
                  4: self._packet.read4ByteTxRx}[length]
        value, result, error = method(self._port, int(self.cfg.hardware["motor_id"]), address)
        if result != self._sdk.COMM_SUCCESS or error:
            raise RuntimeError(f"read 실패: {name}, comm={result}, error={error}")
        return _signed(int(value), length * 8) if signed else int(value)

    def write(self, name: str, value: int) -> None:
        self._require()
        address, length, signed = ADDR[name]
        encoded = int(value) & ((1 << (8 * length)) - 1) if signed else int(value)
        method = {1: self._packet.write1ByteTxRx, 2: self._packet.write2ByteTxRx,
                  4: self._packet.write4ByteTxRx}[length]
        result, error = method(self._port, int(self.cfg.hardware["motor_id"]), address, encoded)
        if result != self._sdk.COMM_SUCCESS or error:
            raise RuntimeError(f"write 실패: {name}, comm={result}, error={error}")

    def torque(self, enabled: bool) -> None:
        self.write("torque_enable", 1 if enabled else 0)

    def configure_and_verify(self) -> dict[str, int]:
        self.torque(False)
        self.write("bus_watchdog_raw", 0)
        desired = {
            "drive_mode": int(self.cfg.registers["drive_mode"]), "operating_mode": 3,
            "position_p_gain": 850, "position_i_gain": int(self.cfg.registers["position_i_gain"]),
            "position_d_gain": int(self.cfg.registers["position_d_gain"]),
            "feedforward_1st_gain": int(self.cfg.registers["feedforward_1st_gain"]),
            "feedforward_2nd_gain": int(self.cfg.registers["feedforward_2nd_gain"]),
            "profile_velocity": int(self.cfg.registers["profile_velocity"]),
            "profile_acceleration": int(self.cfg.registers["profile_acceleration"]),
        }
        for name, value in desired.items():
            if self.read(name) != value:
                self.write(name, value)
        expected = {
            **desired,
            "pwm_limit_raw": int(self.cfg.registers["expected_pwm_limit_raw"]),
            "current_limit_raw": int(self.cfg.registers["expected_current_limit_raw"]),
            "homing_offset_raw": int(self.cfg.hardware["expected_homing_offset_raw"]),
            "bus_watchdog_raw": 0,
        }
        actual = {name: self.read(name) for name in expected}
        mismatch = {name: (expected[name], actual[name]) for name in expected if expected[name] != actual[name]}
        if mismatch:
            raise RuntimeError(f"Mode 3 register read-back 불일치: {mismatch}")
        return actual

    def arm_bus_watchdog(self) -> int:
        value = int(self.cfg.registers["bus_watchdog_raw"])
        self.write("bus_watchdog_raw", value)
        if self.read("bus_watchdog_raw") != value:
            raise RuntimeError("Bus Watchdog readback mismatch")
        return value

    def read_configuration_snapshot(self) -> dict[str, int]:
        names = (
            "firmware_version", "return_delay_time_raw", "homing_offset_raw", "drive_mode",
            "operating_mode", "pwm_limit_raw", "current_limit_raw", "position_d_gain",
            "position_i_gain", "position_p_gain", "feedforward_2nd_gain",
            "feedforward_1st_gain", "bus_watchdog_raw", "profile_acceleration",
            "profile_velocity", "status_return_level_raw", "hardware_error",
        )
        return {name: self.read(name) for name in names}

    def write_goal_rad(self, angle: float) -> int:
        raw = self.cfg.rad_to_raw(angle)
        self.write("goal_position_raw", raw)
        return raw

    def write_goal_rad_no_response(self, angle: float) -> GoalWrite:
        self._require()
        if self._goal_writer is None:
            raise RuntimeError("Goal Position GroupSyncWrite가 초기화되지 않았습니다.")
        raw = self.cfg.rad_to_raw(angle)
        encoded = raw & 0xFFFFFFFF
        data = [(encoded >> shift) & 0xFF for shift in (0, 8, 16, 24)]
        self._goal_writer.clearParam()
        if not self._goal_writer.addParam(int(self.cfg.hardware["motor_id"]), data):
            raise RuntimeError("Goal Position GroupSyncWrite addParam 실패")
        tx_before_ns = time.monotonic_ns()
        result = self._goal_writer.txPacket()
        tx_after_ns = time.monotonic_ns()
        if result != self._sdk.COMM_SUCCESS:
            raise RuntimeError(f"Goal Position syncWriteTxOnly 실패: {self._packet.getTxRxResult(result)}")
        return GoalWrite(raw, tx_before_ns, tx_after_ns)

    def read_state(self) -> State:
        return self.read_state_timed().state

    def read_state_timed(self) -> TimedState:
        self._require()
        read_before_ns = time.monotonic_ns()
        result = self._reader.txRxPacket()
        read_after_ns = time.monotonic_ns()
        if result != self._sdk.COMM_SUCCESS:
            raise RuntimeError(self._packet.getTxRxResult(result))
        motor_id = int(self.cfg.hardware["motor_id"])

        def get(address: int, length: int, signed: bool = False) -> int:
            if not self._reader.isAvailable(motor_id, address, length):
                raise RuntimeError(f"state 데이터 누락: addr={address}")
            value = int(self._reader.getData(motor_id, address, length))
            return _signed(value, length * 8) if signed else value

        state = State(
            goal_position_raw=get(116, 4, True), realtime_tick_raw=get(120, 2),
            moving=get(122, 1), moving_status=get(123, 1), present_pwm_raw=get(124, 2, True),
            present_current_raw=get(126, 2, True), present_velocity_raw=get(128, 4, True),
            present_position_raw=get(132, 4, True), velocity_trajectory_raw=get(136, 4, True),
            position_trajectory_raw=get(140, 4, True), input_voltage_raw=get(144, 2),
            temperature_c=get(146, 1),
        )
        return TimedState(state, read_before_ns, read_after_ns)

    def read_hardware_error(self) -> int:
        return self.read("hardware_error")


__all__ = ["Mode3Bus", "State", "TimedState", "GoalWrite"]
