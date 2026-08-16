from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import RobotConfig


ADDR_HARDWARE_ERROR = 70
ADDR_TORQUE_ENABLE = 64
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84
ADDR_GOAL_POSITION = 116
ADDR_STATE_START = 120
LEN_GOAL_POSITION = 4
LEN_STATE_BLOCK = 27  # 120 Realtime Tick ... 146 Temperature

# Real2Sim run을 재현하기 위해 Torque On 전에 저장할 MX-106(2.0) 설정이다.
# (name, address, byte length, signed)
ACTUATOR_SETTING_REGISTERS = (
    ("firmware_version", 6, 1, False),
    ("drive_mode", 10, 1, False),
    ("operating_mode", 11, 1, False),
    ("homing_offset", 20, 4, True),
    ("temperature_limit", 31, 1, False),
    ("max_voltage_limit_raw", 32, 2, False),
    ("min_voltage_limit_raw", 34, 2, False),
    ("pwm_limit_raw", 36, 2, False),
    ("max_position_limit_tick", 48, 4, False),
    ("min_position_limit_tick", 52, 4, False),
    ("shutdown", 63, 1, False),
    ("status_return_level", 68, 1, False),
    ("velocity_i_gain", 76, 2, False),
    ("velocity_p_gain", 78, 2, False),
    ("position_d_gain", 80, 2, False),
    ("position_i_gain", 82, 2, False),
    ("position_p_gain", 84, 2, False),
    ("feedforward_2nd_gain", 88, 2, False),
    ("feedforward_1st_gain", 90, 2, False),
    ("bus_watchdog_raw", 98, 1, True),
    ("goal_pwm_raw", 100, 2, True),
    ("goal_current_raw", 102, 2, True),
    ("profile_acceleration_raw", 108, 4, False),
    ("profile_velocity_raw", 112, 4, False),
)


def _signed(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def _u32_bytes(value: int) -> list[int]:
    value &= 0xFFFFFFFF
    return [(value >> shift) & 0xFF for shift in (0, 8, 16, 24)]


@dataclass(frozen=True)
class MotorState:
    present_position_tick: int
    present_velocity_raw: int
    present_pwm_raw: int
    present_current_raw: int
    position_trajectory_tick: int
    velocity_trajectory_raw: int
    realtime_tick_ms: int
    input_voltage_raw: int
    temperature_c: int
    moving: int
    moving_status: int


class DynamixelBus:
    """Protocol 2.0 sync I/O. 객체 생성만으로는 포트를 열거나 Torque On하지 않는다."""

    def __init__(self, config: RobotConfig):
        config.require_hardware_ready()
        self.config = config
        self._sdk: Any | None = None
        self._port: Any | None = None
        self._packet: Any | None = None
        self._writer: Any | None = None
        self._state_reader: Any | None = None
        self._error_reader: Any | None = None

    @property
    def motor_ids(self) -> tuple[int, ...]:
        return tuple(int(joint.motor_id) for joint in self.config.joints if joint.motor_id is not None)

    def open(self) -> None:
        if self._port is not None:
            return
        try:
            import dynamixel_sdk as sdk
        except ImportError as exc:
            raise RuntimeError("dynamixel-sdk가 설치되지 않았습니다. uv sync를 실행하세요.") from exc
        self._sdk = sdk
        self._port = sdk.PortHandler(self.config.bus.port)
        self._packet = sdk.PacketHandler(self.config.bus.protocol_version)
        if not self._port.openPort():
            raise RuntimeError(f"포트를 열 수 없습니다: {self.config.bus.port}")
        if not self._port.setBaudRate(self.config.bus.baudrate):
            self._port.closePort()
            self._port = None
            raise RuntimeError(f"baudrate 설정 실패: {self.config.bus.baudrate}")
        self._writer = sdk.GroupSyncWrite(
            self._port, self._packet, ADDR_GOAL_POSITION, LEN_GOAL_POSITION
        )
        self._state_reader = sdk.GroupSyncRead(
            self._port, self._packet, ADDR_STATE_START, LEN_STATE_BLOCK
        )
        self._error_reader = sdk.GroupSyncRead(
            self._port, self._packet, ADDR_HARDWARE_ERROR, 1
        )
        for motor_id in self.motor_ids:
            if not self._state_reader.addParam(motor_id):
                raise RuntimeError(f"state SyncRead addParam 실패: ID {motor_id}")
            if not self._error_reader.addParam(motor_id):
                raise RuntimeError(f"error SyncRead addParam 실패: ID {motor_id}")

    def close(self) -> None:
        if self._port is not None:
            self._port.closePort()
        self._port = None

    def __enter__(self) -> "DynamixelBus":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._port is None or self._packet is None or self._sdk is None:
            raise RuntimeError("Dynamixel 포트가 열리지 않았습니다.")

    def ping_all(self) -> dict[int, int]:
        self._require_open()
        models = {}
        for motor_id in self.motor_ids:
            model, result, error = self._packet.ping(self._port, motor_id)
            if result != self._sdk.COMM_SUCCESS or error:
                raise RuntimeError(f"Ping 실패 ID={motor_id}, comm={result}, error={error}")
            models[motor_id] = model
        return models

    def _read_register(self, motor_id: int, address: int, length: int) -> int:
        self._require_open()
        method = {
            1: self._packet.read1ByteTxRx,
            2: self._packet.read2ByteTxRx,
            4: self._packet.read4ByteTxRx,
        }.get(length)
        if method is None:
            raise ValueError(f"지원하지 않는 register 길이: {length}")
        value, result, error = method(self._port, motor_id, address)
        if result != self._sdk.COMM_SUCCESS or error:
            raise RuntimeError(
                f"Register read 실패 ID={motor_id}, addr={address}, "
                f"comm={result}, error={error}"
            )
        return int(value)

    def _write_register(self, motor_id: int, address: int, length: int, value: int) -> None:
        self._require_open()
        if not 0 <= value < (1 << (8 * length)):
            raise ValueError(
                f"register 값 범위 초과: addr={address}, length={length}, value={value}"
            )
        method = {
            1: self._packet.write1ByteTxRx,
            2: self._packet.write2ByteTxRx,
            4: self._packet.write4ByteTxRx,
        }.get(length)
        if method is None:
            raise ValueError(f"지원하지 않는 register 길이: {length}")
        result, error = method(self._port, motor_id, address, value)
        if result != self._sdk.COMM_SUCCESS or error:
            raise RuntimeError(
                f"Register write 실패 ID={motor_id}, addr={address}, "
                f"comm={result}, error={error}"
            )

    def write_position_pid_gains(
        self,
        *,
        p_gain: int | None = None,
        i_gain: int | None = None,
        d_gain: int | None = None,
    ) -> dict[int, dict[str, int]]:
        """Torque Off 상태에서 요청한 RAM PID만 쓰고 전 모터 readback을 검증한다."""
        requested = {
            "position_p_gain": (ADDR_POSITION_P_GAIN, p_gain),
            "position_i_gain": (ADDR_POSITION_I_GAIN, i_gain),
            "position_d_gain": (ADDR_POSITION_D_GAIN, d_gain),
        }
        requested = {
            name: (address, int(value))
            for name, (address, value) in requested.items()
            if value is not None
        }
        if not requested:
            return {}
        for name, (_, value) in requested.items():
            if not 0 <= value <= 16383:
                raise ValueError(f"{name}={value}: MX position gain 범위 [0,16383] 밖입니다.")
        per_motor = {
            motor_id: {
                name: value for name, (_, value) in requested.items()
            }
            for motor_id in self.motor_ids
        }
        return self.write_position_pid_gains_by_motor(per_motor)

    def write_position_pid_gains_by_motor(
        self, gains: Mapping[int, Mapping[str, int]]
    ) -> dict[int, dict[str, int]]:
        """모터별 position_p/i/d_gain을 쓰고 전체 설정 readback을 검증한다."""
        if set(gains) != set(self.motor_ids):
            raise ValueError(
                f"PID motor ID 불일치: expected={self.motor_ids}, actual={tuple(gains)}"
            )
        addresses = {
            "position_p_gain": ADDR_POSITION_P_GAIN,
            "position_i_gain": ADDR_POSITION_I_GAIN,
            "position_d_gain": ADDR_POSITION_D_GAIN,
        }
        for motor_id, values in gains.items():
            if not values:
                raise ValueError(f"ID {motor_id}: PID 값이 비어 있습니다.")
            unknown = set(values) - set(addresses)
            if unknown:
                raise ValueError(f"ID {motor_id}: 알 수 없는 PID 항목 {unknown}")
            for name, value in values.items():
                value = int(value)
                if not 0 <= value <= 16383:
                    raise ValueError(f"ID {motor_id}: {name}={value} 범위 초과")
                self._write_register(motor_id, addresses[name], 2, value)
        settings = self.read_actuator_settings()
        mismatches = {
            motor_id: {
                name: {"expected": int(value), "actual": settings[motor_id][name]}
                for name, value in values.items()
                if settings[motor_id][name] != int(value)
            }
            for motor_id, values in gains.items()
        }
        mismatches = {key: value for key, value in mismatches.items() if value}
        if mismatches:
            raise RuntimeError(f"Position PID readback 불일치: {mismatches}")
        return settings

    def read_actuator_settings(self) -> dict[int, dict[str, int]]:
        """각 모터의 식별 재현 필수 설정을 Torque Off 상태에서 읽는다."""
        settings: dict[int, dict[str, int]] = {}
        for motor_id in self.motor_ids:
            values: dict[str, int] = {}
            for name, address, length, signed in ACTUATOR_SETTING_REGISTERS:
                value = self._read_register(motor_id, address, length)
                values[name] = _signed(value, length * 8) if signed else value
            settings[motor_id] = values
        return settings

    def set_torque(self, enabled: bool) -> None:
        self._require_open()
        value = 1 if enabled else 0
        failures = []
        for motor_id in self.motor_ids:
            result, error = self._packet.write1ByteTxRx(
                self._port, motor_id, ADDR_TORQUE_ENABLE, value
            )
            if result != self._sdk.COMM_SUCCESS or error:
                failures.append((motor_id, result, error))
        if failures:
            raise RuntimeError(f"Torque {'On' if enabled else 'Off'} 실패: {failures}")

    def write_goal_ticks(self, goals: Mapping[int, int]) -> None:
        self._require_open()
        self._writer.clearParam()
        for motor_id in self.motor_ids:
            if motor_id not in goals:
                raise ValueError(f"Goal Position 누락: ID {motor_id}")
            if not self._writer.addParam(motor_id, _u32_bytes(int(goals[motor_id]))):
                raise RuntimeError(f"SyncWrite addParam 실패: ID {motor_id}")
        result = self._writer.txPacket()
        self._writer.clearParam()
        if result != self._sdk.COMM_SUCCESS:
            raise RuntimeError(self._packet.getTxRxResult(result))

    def read_state(self) -> dict[int, MotorState]:
        self._require_open()
        result = self._state_reader.txRxPacket()
        if result != self._sdk.COMM_SUCCESS:
            raise RuntimeError(self._packet.getTxRxResult(result))
        states = {}
        for motor_id in self.motor_ids:
            def get(address: int, length: int) -> int:
                if not self._state_reader.isAvailable(motor_id, address, length):
                    raise RuntimeError(f"SyncRead 데이터 누락: ID={motor_id}, addr={address}")
                return int(self._state_reader.getData(motor_id, address, length))

            states[motor_id] = MotorState(
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
        return states

    def read_hardware_errors(self) -> dict[int, int]:
        """주소 70만 별도 SyncRead한다. 100 Hz 상태 읽기와 같은 주기에 호출하지 않는다."""
        self._require_open()
        result = self._error_reader.txRxPacket()
        if result != self._sdk.COMM_SUCCESS:
            raise RuntimeError(self._packet.getTxRxResult(result))
        errors = {}
        for motor_id in self.motor_ids:
            if not self._error_reader.isAvailable(motor_id, ADDR_HARDWARE_ERROR, 1):
                raise RuntimeError(f"Hardware Error 데이터 누락: ID={motor_id}")
            errors[motor_id] = int(
                self._error_reader.getData(motor_id, ADDR_HARDWARE_ERROR, 1)
            )
        return errors
