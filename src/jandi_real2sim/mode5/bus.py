from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Mode5Campaign


ADDR = {
    "firmware_version": (6, 1, False),
    "operating_mode": (11, 1, False),
    "homing_offset": (20, 4, True),
    "pwm_limit_raw": (36, 2, False),
    "current_limit_raw": (38, 2, False),
    "shutdown": (63, 1, False),
    "torque_enable": (64, 1, False),
    "hardware_error": (70, 1, False),
    "position_d_gain": (80, 2, False),
    "position_i_gain": (82, 2, False),
    "position_p_gain": (84, 2, False),
    "feedforward_2nd_gain": (88, 2, False),
    "feedforward_1st_gain": (90, 2, False),
    "bus_watchdog_raw": (98, 1, True),
    "goal_pwm_raw": (100, 2, True),
    "goal_current_raw": (102, 2, True),
    "profile_acceleration": (108, 4, False),
    "profile_velocity": (112, 4, False),
    "goal_position": (116, 4, True),
}


def _signed(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def _encoded(value: int, length: int) -> int:
    return value & ((1 << (8 * length)) - 1)


@dataclass(frozen=True)
class State:
    realtime_tick_ms: int
    moving: int
    moving_status: int
    present_pwm_raw: int
    present_current_raw: int
    present_velocity_raw: int
    present_position_tick: int
    velocity_trajectory_raw: int
    position_trajectory_tick: int
    input_voltage_raw: int
    temperature_c: int


class Mode5Bus:
    """한 개 MX-106R(2.0)용 Protocol 2.0 I/O."""

    def __init__(self, cfg: Mode5Campaign):
        self.cfg = cfg
        self._sdk: Any | None = None
        self._port: Any | None = None
        self._packet: Any | None = None
        self._state_reader: Any | None = None

    def open(self) -> None:
        if self._port is not None:
            return
        import dynamixel_sdk as sdk

        self._sdk = sdk
        self._port = sdk.PortHandler(self.cfg.hardware.port)
        self._packet = sdk.PacketHandler(self.cfg.hardware.protocol_version)
        if not self._port.openPort():
            raise RuntimeError(f"포트를 열 수 없습니다: {self.cfg.hardware.port}")
        if not self._port.setBaudRate(self.cfg.hardware.baudrate):
            self.close()
            raise RuntimeError(f"baudrate 설정 실패: {self.cfg.hardware.baudrate}")
        self._state_reader = sdk.GroupSyncRead(self._port, self._packet, 120, 27)
        if not self._state_reader.addParam(self.cfg.hardware.motor_id):
            self.close()
            raise RuntimeError("state GroupSyncRead addParam 실패")

    def close(self) -> None:
        if self._port is not None:
            self._port.closePort()
        self._port = None

    def __enter__(self) -> "Mode5Bus":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _require(self) -> None:
        if self._port is None or self._packet is None or self._sdk is None:
            raise RuntimeError("DYNAMIXEL 포트가 열리지 않았습니다.")

    def ping(self) -> int:
        self._require()
        model, result, error = self._packet.ping(
            self._port, self.cfg.hardware.motor_id
        )
        if result != self._sdk.COMM_SUCCESS or error:
            raise RuntimeError(f"Ping 실패: comm={result}, error={error}")
        return int(model)

    def read(self, name: str) -> int:
        self._require()
        address, length, signed = ADDR[name]
        method = {
            1: self._packet.read1ByteTxRx,
            2: self._packet.read2ByteTxRx,
            4: self._packet.read4ByteTxRx,
        }[length]
        value, result, error = method(self._port, self.cfg.hardware.motor_id, address)
        if result != self._sdk.COMM_SUCCESS or error:
            raise RuntimeError(
                f"read 실패: {name}, comm={result}, error={error}"
            )
        value = int(value)
        return _signed(value, length * 8) if signed else value

    def write(self, name: str, value: int) -> None:
        self._require()
        address, length, signed = ADDR[name]
        if signed:
            value = _encoded(int(value), length)
        elif not 0 <= int(value) < 1 << (8 * length):
            raise ValueError(f"{name} register 범위 초과: {value}")
        method = {
            1: self._packet.write1ByteTxRx,
            2: self._packet.write2ByteTxRx,
            4: self._packet.write4ByteTxRx,
        }[length]
        result, error = method(
            self._port, self.cfg.hardware.motor_id, address, int(value)
        )
        if result != self._sdk.COMM_SUCCESS or error:
            raise RuntimeError(
                f"write 실패: {name}, comm={result}, error={error}"
            )

    def torque(self, enabled: bool) -> None:
        self.write("torque_enable", 1 if enabled else 0)

    def read_settings(self) -> dict[str, int]:
        names = tuple(name for name in ADDR if name != "goal_position")
        return {name: self.read(name) for name in names}

    def configure_and_verify(self) -> dict[str, int]:
        """Torque Off → Mode 5 → 고정 register 기록 → 전 항목 read-back."""
        registers = self.cfg.registers
        self.torque(False)
        if self.read("operating_mode") != 5:
            self.write("operating_mode", 5)
        writes = {
            "position_p_gain": registers.position_p_gain,
            "position_i_gain": registers.position_i_gain,
            "position_d_gain": registers.position_d_gain,
            "feedforward_1st_gain": registers.feedforward_1st_gain,
            "feedforward_2nd_gain": registers.feedforward_2nd_gain,
            "profile_velocity": registers.profile_velocity,
            "profile_acceleration": registers.profile_acceleration,
            "goal_current_raw": registers.goal_current_raw,
            "goal_pwm_raw": registers.goal_pwm_raw,
            "bus_watchdog_raw": registers.bus_watchdog_raw,
        }
        if any(value is None for value in writes.values()):
            raise RuntimeError("미확정 Mode 5 register가 있어 실기체 설정을 중단했습니다.")
        for name, value in writes.items():
            assert value is not None
            self.write(name, int(value))

        actual = self.read_settings()
        expected = {"operating_mode": 5, **{k: int(v) for k, v in writes.items()}}
        expected["current_limit_raw"] = int(registers.expected_current_limit_raw)  # type: ignore[arg-type]
        expected["pwm_limit_raw"] = int(registers.expected_pwm_limit_raw)  # type: ignore[arg-type]
        mismatches = {
            name: {"expected": value, "actual": actual[name]}
            for name, value in expected.items()
            if actual[name] != value
        }
        if mismatches:
            raise RuntimeError(f"Mode 5 register read-back 불일치: {mismatches}")
        return actual

    def write_goal_rad(self, goal_rad: float) -> None:
        self.write("goal_position", self.cfg.rad_to_tick(goal_rad))

    def read_state(self) -> State:
        self._require()
        result = self._state_reader.txRxPacket()
        if result != self._sdk.COMM_SUCCESS:
            raise RuntimeError(self._packet.getTxRxResult(result))
        motor_id = self.cfg.hardware.motor_id

        def get(address: int, length: int) -> int:
            if not self._state_reader.isAvailable(motor_id, address, length):
                raise RuntimeError(f"state 데이터 누락: addr={address}")
            return int(self._state_reader.getData(motor_id, address, length))

        return State(
            realtime_tick_ms=get(120, 2),
            moving=get(122, 1),
            moving_status=get(123, 1),
            present_pwm_raw=_signed(get(124, 2), 16),
            present_current_raw=_signed(get(126, 2), 16),
            present_velocity_raw=_signed(get(128, 4), 32),
            present_position_tick=_signed(get(132, 4), 32),
            velocity_trajectory_raw=_signed(get(136, 4), 32),
            position_trajectory_tick=_signed(get(140, 4), 32),
            input_voltage_raw=get(144, 2),
            temperature_c=get(146, 1),
        )

    def read_hardware_error(self) -> int:
        return self.read("hardware_error")
