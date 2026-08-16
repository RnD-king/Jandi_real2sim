from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


TRAJECTORIES = ("step", "triangle", "sine")
CONDITIONS = ("no_load", "loaded")


def _need(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"필수 항목이 없습니다: {key}")
    return mapping[key]


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name}은 0보다 커야 합니다.")
    return result


@dataclass(frozen=True)
class Hardware:
    port: str
    baudrate: int
    protocol_version: float
    motor_id: int
    joint_name: str
    zero_tick: int
    direction: int
    min_rad: float
    max_rad: float
    center_rad: float


@dataclass(frozen=True)
class Timing:
    command_rate_hz: int
    state_read_rate_hz: int
    hardware_error_read_rate_hz: int
    transition_sec: float


@dataclass(frozen=True)
class Registers:
    operating_mode: int
    position_p_gain: int | None
    position_i_gain: int | None
    position_d_gain: int | None
    feedforward_1st_gain: int | None
    feedforward_2nd_gain: int | None
    profile_velocity: int | None
    profile_acceleration: int | None
    goal_current_raw: int | None
    expected_current_limit_raw: int | None
    goal_pwm_raw: int | None
    expected_pwm_limit_raw: int | None
    bus_watchdog_raw: int | None

    def unresolved(self) -> list[str]:
        return [
            name
            for name, value in vars(self).items()
            if name != "operating_mode" and value is None
        ]


@dataclass(frozen=True)
class Bench:
    bare_horn: bool
    gravity_zero_offset_rad: float | None
    arm_mass_kg: float | None
    arm_com_radius_m: float | None
    arm_inertia_kg_m2: float | None
    added_load_mass_kg: float | None
    added_load_radius_m: float | None

    @property
    def equivalent_mass_kg(self) -> float | None:
        if self.arm_mass_kg is None or self.added_load_mass_kg is None:
            return None
        return self.arm_mass_kg + self.added_load_mass_kg

    @property
    def equivalent_com_radius_m(self) -> float | None:
        mass = self.equivalent_mass_kg
        if (
            mass is None
            or mass <= 0.0
            or self.arm_com_radius_m is None
            or self.added_load_radius_m is None
        ):
            return None
        return (
            self.arm_mass_kg * self.arm_com_radius_m
            + self.added_load_mass_kg * self.added_load_radius_m
        ) / mass

    @property
    def equivalent_pivot_inertia_kg_m2(self) -> float | None:
        if (
            self.arm_inertia_kg_m2 is None
            or self.added_load_mass_kg is None
            or self.added_load_radius_m is None
        ):
            return None
        return (
            self.arm_inertia_kg_m2
            + self.added_load_mass_kg * self.added_load_radius_m**2
        )

    def resolved_metadata(self) -> dict[str, float | None]:
        return {
            **vars(self),
            "equivalent_mass_kg": self.equivalent_mass_kg,
            "equivalent_com_radius_m": self.equivalent_com_radius_m,
            "equivalent_pivot_inertia_kg_m2": self.equivalent_pivot_inertia_kg_m2,
        }

    def unresolved(self, condition: str) -> list[str]:
        if condition == "no_load" and self.bare_horn:
            return []
        required = (
            "gravity_zero_offset_rad",
            "arm_mass_kg",
            "arm_com_radius_m",
            "arm_inertia_kg_m2",
            "added_load_mass_kg",
            "added_load_radius_m",
        )
        return [name for name in required if getattr(self, name) is None]


@dataclass(frozen=True)
class Safety:
    pilot_approved: bool
    pilot_amplitude_rad: float
    max_temperature_c: float
    min_input_voltage_v: float
    max_abs_current_a: float
    max_abs_pwm_percent: float
    max_abs_position_error_rad: float
    consecutive_state_samples: int
    between_runs_sec: float


@dataclass(frozen=True)
class Mode5Campaign:
    source: Path
    project_root: Path
    source_files: dict[str, Path]
    campaign_id: str
    output_root: Path
    hardware: Hardware
    timing: Timing
    registers: Registers
    benches: dict[str, Bench]
    trajectories: dict[str, dict[str, Any]]
    repeats: tuple[int, ...]
    safety: Safety

    def unresolved_for_execution(
        self,
        *,
        condition: str | None = None,
        require_pilot_approval: bool = True,
    ) -> list[str]:
        unresolved = [f"mode5.{name}" for name in self.registers.unresolved()]
        selected = CONDITIONS if condition is None else (condition,)
        for condition_name in selected:
            unresolved.extend(
                f"conditions.{condition_name}.{name}"
                for name in self.benches[condition_name].unresolved(condition_name)
            )
        if require_pilot_approval and not self.safety.pilot_approved:
            unresolved.append("safety.pilot_approved")
        return unresolved

    def validate_motion(self, angle_rad: float) -> None:
        if not self.hardware.min_rad <= angle_rad <= self.hardware.max_rad:
            raise ValueError(
                f"{angle_rad:+.6f} rad가 software limit "
                f"[{self.hardware.min_rad:+.6f}, {self.hardware.max_rad:+.6f}] 밖입니다."
            )

    def rad_to_tick(self, angle_rad: float) -> int:
        self.validate_motion(angle_rad)
        tick = round(
            self.hardware.zero_tick
            + self.hardware.direction * angle_rad * 4096 / (2.0 * 3.141592653589793)
        )
        if not 0 <= tick <= 4095:
            raise ValueError(f"Goal Position {tick} tick이 [0,4095] 밖입니다.")
        return tick

    def tick_to_rad(self, tick: int) -> float:
        return (
            self.hardware.direction
            * (tick - self.hardware.zero_tick)
            * 2.0
            * 3.141592653589793
            / 4096
        )


def _bench(raw: dict[str, Any]) -> Bench:
    def optional(name: str) -> float | None:
        value = raw.get(name)
        return None if value is None else float(value)

    return Bench(
        bare_horn=bool(raw.get("bare_horn", False)),
        gravity_zero_offset_rad=optional("gravity_zero_offset_rad"),
        arm_mass_kg=optional("arm_mass_kg"),
        arm_com_radius_m=optional("arm_com_radius_m"),
        arm_inertia_kg_m2=optional("arm_inertia_kg_m2"),
        added_load_mass_kg=optional("added_load_mass_kg"),
        added_load_radius_m=optional("added_load_radius_m"),
    )


def _project_root(source: Path) -> Path:
    for parent in (source.parent, *source.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError(f"pyproject.toml을 기준으로 프로젝트 루트를 찾지 못했습니다: {source}")


def _yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"YAML 최상위는 mapping이어야 합니다: {path}")
    return raw


def _component(source: Path, relative: Any, role: str, files: dict[str, Path]) -> dict[str, Any]:
    path = (source.parent / str(relative)).resolve()
    if not path.is_file():
        raise ValueError(f"{role} 구성 파일이 없습니다: {path}")
    files[role] = path
    raw = _yaml(path)
    if int(_need(raw, "schema_version")) != 1:
        raise ValueError(f"{role} 구성의 schema_version은 1이어야 합니다: {path}")
    return raw


def _compose_v2(source: Path, raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Path]]:
    components = _need(raw, "components")
    files = {"campaign": source}
    hardware = _component(source, _need(components, "hardware"), "hardware", files)
    controller = _component(source, _need(components, "controller"), "controller", files)

    condition_refs = _need(components, "conditions")
    conditions: dict[str, Any] = {}
    for name in CONDITIONS:
        condition = _component(
            source, _need(condition_refs, name), f"condition.{name}", files
        )
        if condition.get("condition") != name:
            raise ValueError(f"condition.{name} 파일 내부 condition 이름이 일치하지 않습니다.")
        if "base" in condition:
            base = _component(
                files[f"condition.{name}"],
                condition["base"],
                f"condition.{name}.base",
                files,
            )
            merged = dict(_need(base, "bench"))
            merged.update(_need(condition, "bench"))
            conditions[name] = merged
        else:
            conditions[name] = dict(_need(condition, "bench"))

    trajectory_refs = _need(components, "trajectories")
    trajectories: dict[str, Any] = {}
    for name in TRAJECTORIES:
        trajectory = _component(
            source, _need(trajectory_refs, name), f"trajectory.{name}", files
        )
        if trajectory.get("trajectory") != name:
            raise ValueError(f"trajectory.{name} 파일 내부 trajectory 이름이 일치하지 않습니다.")
        trajectories[name] = dict(_need(trajectory, "parameters"))

    return {
        "schema_version": 1,
        "campaign": _need(raw, "campaign"),
        "hardware": _need(hardware, "hardware"),
        "timing": _need(hardware, "timing"),
        "mode5_registers": _need(controller, "mode5_registers"),
        "conditions": conditions,
        "trajectories": trajectories,
        "repeats": _need(raw, "repeats"),
        "safety": _need(raw, "safety"),
    }, files


def load_campaign(path: str | Path) -> Mode5Campaign:
    source = Path(path).expanduser().resolve()
    raw = _yaml(source)
    if "redirect" in raw:
        return load_campaign((source.parent / str(raw["redirect"])).resolve())
    schema_version = int(_need(raw, "schema_version"))
    if schema_version == 2:
        raw, source_files = _compose_v2(source, raw)
    elif schema_version == 1:
        source_files = {"campaign": source}
    else:
        raise ValueError("지원하는 mode5 campaign schema_version은 1 또는 2입니다.")
    project_root = _project_root(source)
    campaign_raw = _need(raw, "campaign")
    hw = _need(raw, "hardware")
    timing = _need(raw, "timing")
    registers = _need(raw, "mode5_registers")
    conditions = _need(raw, "conditions")
    trajectories = _need(raw, "trajectories")
    safety = _need(raw, "safety")

    if tuple(conditions) != CONDITIONS:
        raise ValueError(f"conditions 순서는 정확히 {CONDITIONS}여야 합니다.")
    if tuple(trajectories) != TRAJECTORIES:
        raise ValueError(f"trajectories 순서는 정확히 {TRAJECTORIES}여야 합니다.")

    cfg = Mode5Campaign(
        source=source,
        project_root=project_root,
        source_files=source_files,
        campaign_id=str(_need(campaign_raw, "id")),
        output_root=(project_root / str(_need(campaign_raw, "output_root"))).resolve(),
        hardware=Hardware(
            port=str(_need(hw, "port")),
            baudrate=int(_need(hw, "baudrate")),
            protocol_version=float(_need(hw, "protocol_version")),
            motor_id=int(_need(hw, "motor_id")),
            joint_name=str(_need(hw, "joint_name")),
            zero_tick=int(_need(hw, "zero_tick")),
            direction=int(_need(hw, "direction")),
            min_rad=float(_need(hw, "software_min_rad")),
            max_rad=float(_need(hw, "software_max_rad")),
            center_rad=float(_need(hw, "center_rad")),
        ),
        timing=Timing(
            command_rate_hz=int(_need(timing, "command_rate_hz")),
            state_read_rate_hz=int(_need(timing, "state_read_rate_hz")),
            hardware_error_read_rate_hz=int(
                _need(timing, "hardware_error_read_rate_hz")
            ),
            transition_sec=_positive(_need(timing, "transition_sec"), "transition_sec"),
        ),
        registers=Registers(
            operating_mode=int(_need(registers, "operating_mode")),
            position_p_gain=registers.get("position_p_gain"),
            position_i_gain=registers.get("position_i_gain"),
            position_d_gain=registers.get("position_d_gain"),
            feedforward_1st_gain=registers.get("feedforward_1st_gain"),
            feedforward_2nd_gain=registers.get("feedforward_2nd_gain"),
            profile_velocity=registers.get("profile_velocity"),
            profile_acceleration=registers.get("profile_acceleration"),
            goal_current_raw=registers.get("goal_current_raw"),
            expected_current_limit_raw=registers.get("expected_current_limit_raw"),
            goal_pwm_raw=registers.get("goal_pwm_raw"),
            expected_pwm_limit_raw=registers.get("expected_pwm_limit_raw"),
            bus_watchdog_raw=registers.get("bus_watchdog_raw"),
        ),
        benches={name: _bench(conditions[name]) for name in CONDITIONS},
        trajectories={name: dict(trajectories[name]) for name in TRAJECTORIES},
        repeats=tuple(int(value) for value in _need(raw, "repeats")),
        safety=Safety(
            pilot_approved=bool(_need(safety, "pilot_approved")),
            pilot_amplitude_rad=_positive(
                _need(safety, "pilot_amplitude_rad"), "pilot_amplitude_rad"
            ),
            max_temperature_c=float(_need(safety, "max_temperature_c")),
            min_input_voltage_v=float(_need(safety, "min_input_voltage_v")),
            max_abs_current_a=_positive(
                _need(safety, "max_abs_current_a"), "max_abs_current_a"
            ),
            max_abs_pwm_percent=_positive(
                _need(safety, "max_abs_pwm_percent"), "max_abs_pwm_percent"
            ),
            max_abs_position_error_rad=_positive(
                _need(safety, "max_abs_position_error_rad"),
                "max_abs_position_error_rad",
            ),
            consecutive_state_samples=int(_need(safety, "consecutive_state_samples")),
            between_runs_sec=float(_need(safety, "between_runs_sec")),
        ),
    )

    if cfg.hardware.direction not in (-1, 1):
        raise ValueError("hardware.direction은 -1 또는 +1이어야 합니다.")
    if cfg.hardware.min_rad >= cfg.hardware.max_rad:
        raise ValueError("software_min_rad는 software_max_rad보다 작아야 합니다.")
    cfg.validate_motion(cfg.hardware.center_rad)
    cfg.validate_motion(cfg.hardware.center_rad + cfg.safety.pilot_amplitude_rad)
    cfg.validate_motion(cfg.hardware.center_rad - cfg.safety.pilot_amplitude_rad)
    if cfg.registers.operating_mode != 5:
        raise ValueError("Current-based Position Control의 operating_mode는 5여야 합니다.")
    gain_names = (
        "position_p_gain",
        "position_i_gain",
        "position_d_gain",
        "feedforward_1st_gain",
        "feedforward_2nd_gain",
    )
    for name in gain_names:
        value = getattr(cfg.registers, name)
        if value is not None and not 0 <= int(value) <= 16383:
            raise ValueError(f"mode5_registers.{name}이 [0,16383] 밖입니다.")
    if (
        cfg.registers.goal_current_raw is not None
        and cfg.registers.expected_current_limit_raw is not None
        and abs(int(cfg.registers.goal_current_raw))
        > int(cfg.registers.expected_current_limit_raw)
    ):
        raise ValueError("abs(goal_current_raw)가 Current Limit보다 큽니다.")
    if (
        cfg.registers.goal_pwm_raw is not None
        and cfg.registers.expected_pwm_limit_raw is not None
        and abs(int(cfg.registers.goal_pwm_raw)) > int(cfg.registers.expected_pwm_limit_raw)
    ):
        raise ValueError("abs(goal_pwm_raw)가 PWM Limit보다 큽니다.")
    if cfg.registers.bus_watchdog_raw is not None and not 1 <= int(cfg.registers.bus_watchdog_raw) <= 127:
        raise ValueError("bus_watchdog_raw는 실험에서 비활성화(0)하지 않고 [1,127]로 둡니다.")
    if cfg.timing.command_rate_hz != 100:
        raise ValueError("이 campaign의 모터 명령 주기는 100 Hz로 고정합니다.")
    if (
        cfg.timing.state_read_rate_hz + cfg.timing.hardware_error_read_rate_hz
        != cfg.timing.command_rate_hz
    ):
        raise ValueError("state + hardware-error read 슬롯 합은 command rate와 같아야 합니다.")
    if cfg.repeats != (1, 2, 3):
        raise ValueError("repeats는 fit 1·2, validation 3인 [1,2,3]으로 고정합니다.")
    if cfg.safety.consecutive_state_samples < 1:
        raise ValueError("consecutive_state_samples는 1 이상이어야 합니다.")
    no_load = cfg.benches["no_load"]
    loaded = cfg.benches["loaded"]
    if not no_load.bare_horn:
        raise ValueError("no_load는 모터+혼만인 bare_horn: true여야 합니다.")
    for name in (
        "gravity_zero_offset_rad",
        "arm_mass_kg",
        "arm_com_radius_m",
        "arm_inertia_kg_m2",
        "added_load_mass_kg",
        "added_load_radius_m",
    ):
        value = getattr(no_load, name)
        if value not in (0, 0.0):
            raise ValueError(f"bare-horn no_load의 {name}는 정확히 0이어야 합니다.")
    if loaded.bare_horn:
        raise ValueError("loaded는 bare_horn: false여야 합니다.")
    for name in ("arm_mass_kg", "arm_com_radius_m", "arm_inertia_kg_m2"):
        value = getattr(loaded, name)
        if value is not None and value <= 0:
            raise ValueError(f"conditions.loaded.{name}는 0보다 커야 합니다.")
    if loaded.added_load_mass_kg is not None and loaded.added_load_mass_kg <= 0:
        raise ValueError("loaded.added_load_mass_kg는 0보다 커야 합니다.")
    if loaded.added_load_radius_m is not None and loaded.added_load_radius_m <= 0:
        raise ValueError("loaded.added_load_radius_m는 0보다 커야 합니다.")
    return cfg
