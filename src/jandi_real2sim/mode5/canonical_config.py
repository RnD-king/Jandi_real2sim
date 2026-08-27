"""Strict README-v3 configuration loader for the canonical Mode-5 bench."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .spec import APPROACH_DIRECTIONS, CONFIRMATIONS, DEFAULT_CAMPAIGN
from .spec import MAIN_TRAJECTORIES, MECHANICAL_CONFIGURATIONS, REPEATS


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"YAML 최상위는 mapping이어야 합니다: {path}")
    if value.get("schema_version") != 3:
        raise ValueError(f"canonical Mode 5 schema_version은 3이어야 합니다: {path}")
    return value


def _project_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError(f"프로젝트 루트를 찾지 못했습니다: {path}")


def _component(campaign: Path, relative: Any, role: str, files: dict[str, Path]) -> dict[str, Any]:
    path = (campaign.parent / str(relative)).resolve()
    if not path.is_file():
        raise ValueError(f"{role} 구성 파일이 없습니다: {path}")
    files[role] = path
    return _yaml(path)


def _missing(mapping: dict[str, Any], fields: Iterable[str], prefix: str) -> list[str]:
    return [f"{prefix}.{name}" for name in fields if mapping.get(name) is None]


@dataclass(frozen=True)
class MechanicalConfiguration:
    id: str
    arm_length: str
    load: str


@dataclass(frozen=True)
class CanonicalCampaign:
    source: Path
    project_root: Path
    source_files: dict[str, Path]
    campaign: dict[str, Any]
    hardware: dict[str, Any]
    timing: dict[str, Any]
    registers: dict[str, Any]
    safety: dict[str, Any]
    approval: dict[str, Any]
    pilot: dict[str, Any]
    geometry: dict[str, Any]
    loads: dict[str, dict[str, Any]]
    trajectories: dict[str, dict[str, Any]]
    configurations: tuple[MechanicalConfiguration, ...]
    holdout_configuration: str | None
    execution_order: Any
    randomization_seed: Any

    @property
    def campaign_id(self) -> str | None:
        value = self.campaign.get("id")
        return None if value is None else str(value)

    @property
    def output_root(self) -> Path:
        return (self.project_root / str(self.campaign["output_root"])).resolve()

    @property
    def processed_root(self) -> Path:
        return (self.project_root / str(self.campaign["processed_root"])).resolve()

    @property
    def results_root(self) -> Path:
        return (self.project_root / str(self.campaign["results_root"])).resolve()

    @property
    def target_generation_rate_hz(self) -> float:
        return float(self.timing["target_generation_rate_hz"])

    @property
    def bus_write_rate_hz(self) -> float:
        return float(self.timing["bus_write_rate_hz"])

    @property
    def state_read_rate_hz(self) -> float:
        return float(self.timing["state_read_rate_hz"])

    @property
    def command_rate_hz(self) -> float:
        """Compatibility alias for trajectory generation; not the bus rate."""
        return self.target_generation_rate_hz

    def configuration(self, name: str) -> MechanicalConfiguration:
        for item in self.configurations:
            if item.id == name:
                return item
        raise ValueError(f"알 수 없는 mechanical configuration: {name}")

    def arm_length_m(self, config_id: str) -> float:
        item = self.configuration(config_id)
        return float(self.geometry["arm_lengths_m"][item.arm_length])

    def load_mass_kg(self, config_id: str) -> float:
        item = self.configuration(config_id)
        return float(self.loads[item.load]["measured_mass_kg"])

    def config_manifest(self) -> dict[str, dict[str, str]]:
        return {
            role: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for role, path in sorted(self.source_files.items())
        }

    def rad_to_raw(self, angle_rad: float) -> int:
        lo = self.safety.get("software_position_min_rad")
        hi = self.safety.get("software_position_max_rad")
        if lo is None or hi is None:
            raise ValueError("software position limit가 미확정입니다.")
        if not float(lo) <= angle_rad <= float(hi):
            raise ValueError(f"Goal Position {angle_rad:+.6f} rad가 [{lo}, {hi}] 밖입니다.")
        zero = int(self.hardware["encoder_zero_raw"])
        direction = int(self.hardware["direction"])
        raw = round(zero + direction * angle_rad * 4096.0 / (2.0 * 3.141592653589793))
        if not 0 <= raw <= 4095:
            raise ValueError(f"Goal Position raw={raw}가 single-turn 범위 밖입니다.")
        return raw

    def raw_to_rad(self, raw: int) -> float:
        zero = int(self.hardware["encoder_zero_raw"])
        direction = int(self.hardware["direction"])
        return direction * (raw - zero) * 2.0 * 3.141592653589793 / 4096.0

    def common_execution_missing(self) -> list[str]:
        result = _missing(self.campaign, ("id",), "campaign")
        result += _missing(
            self.hardware,
            ("serial_device", "baudrate", "motor_id", "expected_model_number", "encoder_zero_raw", "expected_homing_offset_raw", "direction", "current_direction", "pwm_direction"),
            "hardware",
        )
        result += _missing(
            self.timing,
            ("target_generation_rate_hz", "bus_write_rate_hz", "state_read_rate_hz",
             "hardware_error_poll_rate_hz", "severe_overrun_threshold_sec"),
            "timing",
        )
        result += _missing(
            self.registers,
            ("drive_mode", "position_p_gain", "position_d_gain", "bus_watchdog_raw", "goal_current_raw", "expected_current_limit_raw", "goal_pwm_raw", "expected_pwm_limit_raw"),
            "mode5_registers",
        )
        result += _missing(
            self.safety,
            (
                "software_position_min_rad", "software_position_max_rad", "maximum_temperature_c",
                "minimum_input_voltage_v", "maximum_input_voltage_v", "maximum_abs_current_A",
                "maximum_abs_pwm_fraction", "maximum_abs_position_error_rad",
                "maximum_consecutive_overruns", "oscillation_velocity_limit_rad_s", "transition_duration_sec",
                "between_runs_sec", "warmup_procedure",
            ),
            "safety",
        )
        return result

    def bench_missing(self) -> list[str]:
        result = _missing(
            self.geometry,
            ("arm_mass_kg", "arm_com_radius_m", "arm_inertia_kg_m2", "arm_inertia_reference", "gravity_zero_angle_rad", "gravity_torque_sign"),
            "bench.geometry",
        )
        for name in ("L1", "L2"):
            if self.geometry.get("arm_lengths_m", {}).get(name) is None:
                result.append(f"bench.geometry.arm_lengths_m.{name}")
        for name in ("m250", "m500", "m750"):
            load = self.loads.get(name, {})
            measured = load.get("measured_mass_kg")
            if measured is None:
                result.append(f"bench.loads.{name}.measured_mass_kg")
            else:
                nominal = float(load["nominal_mass_kg"])
                if not 0.5 * nominal <= float(measured) <= 1.5 * nominal:
                    result.append(f"bench.loads.{name}.measured_mass_kg(plausibility)")
        for name, value in self.geometry.get("coordinate", {}).items():
            if value is None:
                result.append(f"bench.geometry.coordinate.{name}")
        return result

    def trajectory_missing(self, name: str) -> list[str]:
        return [f"trajectories.{name}.{key}" for key, value in self.trajectories[name].items() if value is None]

    def execution_missing(self, experiment: str) -> list[str]:
        result = self.common_execution_missing()
        if experiment == "pilot":
            result += _missing(self.pilot, ("mechanical_configuration", "center_rad", "amplitude_rad", "hold_sec"), "pilot")
            selected = self.pilot.get("mechanical_configuration")
            if selected in {item.id for item in self.configurations}:
                load = self.configuration(str(selected)).load
                measured = self.loads[load].get("measured_mass_kg")
                if measured is None:
                    result.append(f"bench.loads.{load}.measured_mass_kg")
                else:
                    nominal = float(self.loads[load]["nominal_mass_kg"])
                    if not 0.5 * nominal <= float(measured) <= 1.5 * nominal:
                        result.append(f"bench.loads.{load}.measured_mass_kg(plausibility)")
            return sorted(set(result))
        result += self.bench_missing()
        if not self.approval.get("pilot_passed", False):
            result.append("approval.pilot_passed")
        if self.approval.get("pilot_run_reference") is None:
            result.append("approval.pilot_run_reference")
        if self.approval.get("pilot_approved_at") is None:
            result.append("approval.pilot_approved_at")
        if self.approval.get("warmup_acknowledged_at") is None:
            result.append("approval.warmup_acknowledged_at")
        if experiment == "static":
            result += self.trajectory_missing("static_calibration")
            if self.execution_order is None:
                result.append("campaign.execution_order")
            if self.randomization_seed is None:
                result.append("campaign.randomization_seed")
        elif experiment == "delay":
            result += self.trajectory_missing("delay_probe")
        elif experiment == "collect":
            for name in MAIN_TRAJECTORIES:
                result += self.trajectory_missing(name)
            if self.holdout_configuration is None:
                result.append("campaign.holdout_configuration")
            if self.execution_order is None:
                result.append("campaign.execution_order")
            if self.randomization_seed is None:
                result.append("campaign.randomization_seed")
        else:
            raise ValueError(f"알 수 없는 experiment: {experiment}")
        return sorted(set(result))


def load_canonical_campaign(path: str | Path = DEFAULT_CAMPAIGN) -> CanonicalCampaign:
    source = Path(path).expanduser().resolve()
    raw = _yaml(source)
    root = _project_root(source)
    files = {"README": root / "README.md", "campaign": source}
    components = raw["components"]
    hardware_raw = _component(source, components["hardware"], "hardware", files)
    controller_raw = _component(source, components["controller"], "controller", files)
    safety_raw = _component(source, components["safety"], "safety", files)
    geometry_raw = _component(source, components["bench_geometry"], "bench.geometry", files)
    loads_raw = _component(source, components["bench_loads"], "bench.loads", files)
    trajectories: dict[str, dict[str, Any]] = {}
    for name in (*MAIN_TRAJECTORIES, "static_calibration", "delay_probe"):
        item = _component(source, components["trajectories"][name], f"trajectory.{name}", files)
        if item.get("trajectory") != name:
            raise ValueError(f"trajectory 파일 이름 불일치: {name}")
        trajectories[name] = dict(item["parameters"])
    configurations = tuple(MechanicalConfiguration(**value) for value in raw["mechanical_configurations"])
    cfg = CanonicalCampaign(
        source=source, project_root=root, source_files=files,
        campaign=dict(raw["campaign"]), hardware=dict(hardware_raw["hardware"]),
        timing=dict(hardware_raw["timing"]), registers=dict(controller_raw["mode5_registers"]),
        safety=dict(safety_raw["safety"]), approval=dict(safety_raw["approval"]),
        pilot=dict(safety_raw["pilot"]),
        geometry={key: value for key, value in geometry_raw.items() if key != "schema_version"},
        loads=dict(loads_raw["loads"]), trajectories=trajectories, configurations=configurations,
        holdout_configuration=raw.get("holdout_configuration"),
        execution_order=raw.get("execution_order"), randomization_seed=raw.get("randomization_seed"),
    )
    _validate_structure(cfg, raw, safety_raw)
    return cfg


def _validate_structure(cfg: CanonicalCampaign, raw: dict[str, Any], safety_raw: dict[str, Any]) -> None:
    ids = tuple(item.id for item in cfg.configurations)
    if ids != MECHANICAL_CONFIGURATIONS:
        raise ValueError(f"mechanical configurations는 정확히 {MECHANICAL_CONFIGURATIONS}여야 합니다.")
    if tuple(raw["main_trajectories"]) != MAIN_TRAJECTORIES:
        raise ValueError(f"main trajectories는 정확히 {MAIN_TRAJECTORIES}여야 합니다.")
    if tuple(raw["repetitions"]) != REPEATS:
        raise ValueError("세 repeat는 repeatability용 [1,2,3]이어야 합니다.")
    if tuple(raw["approach_directions"]) != APPROACH_DIRECTIONS:
        raise ValueError(f"approach directions는 정확히 {APPROACH_DIRECTIONS}여야 합니다.")
    if safety_raw["confirmations"] != CONFIRMATIONS:
        raise ValueError(f"confirmation string은 {CONFIRMATIONS}와 일치해야 합니다.")
    if cfg.registers.get("operating_mode") != 5:
        raise ValueError("Operating Mode는 5여야 합니다.")
    for name in ("position_i_gain", "feedforward_1st_gain", "feedforward_2nd_gain", "profile_velocity", "profile_acceleration"):
        if cfg.registers.get(name) != 0:
            raise ValueError(f"{name}은 canonical controller에서 0이어야 합니다.")
    expected_rates = {
        "target_generation_rate_hz": 50.0,
        "bus_write_rate_hz": 100.0,
        "state_read_rate_hz": 100.0,
    }
    for name, expected in expected_rates.items():
        if float(cfg.timing.get(name, 0)) != expected:
            raise ValueError(f"canonical timing.{name}는 {expected:g} Hz여야 합니다.")
    if cfg.campaign_id is not None and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", cfg.campaign_id) is None:
        raise ValueError("campaign.id는 경로 구분자 없는 영문/숫자/._- 이름이어야 합니다.")
    if cfg.geometry.get("arm_lengths_m") != {"L1": 0.10, "L2": 0.15}:
        raise ValueError("canonical arm lengths는 정확히 L1=0.10 m, L2=0.15 m여야 합니다.")
    expected_static = [-math.pi / 2, -math.pi / 3, -math.pi / 6, 0.0,
                       math.pi / 6, math.pi / 3, math.pi / 2]
    actual_static = cfg.trajectories["static_calibration"].get("static_angles_rad")
    if not isinstance(actual_static, list) or len(actual_static) != 7 or any(
        not math.isclose(float(a), b, rel_tol=0.0, abs_tol=1e-12)
        for a, b in zip(actual_static, expected_static)
    ):
        raise ValueError("canonical static angle set은 정확히 -90,-60,-30,0,+30,+60,+90 deg여야 합니다.")
    if float(cfg.geometry.get("gravity_zero_angle_rad", math.nan)) != 0.0:
        raise ValueError("upright q=0 canonical에서 gravity_zero_angle_rad는 0.0이어야 합니다.")
    if cfg.geometry.get("coordinate", {}).get("zero_definition") != "upright":
        raise ValueError("canonical q=0 coordinate는 physical upright여야 합니다.")
    if cfg.holdout_configuration is not None and cfg.holdout_configuration not in ids:
        raise ValueError("holdout_configuration이 six mechanical configurations에 없습니다.")
    for name in ("direction", "current_direction", "pwm_direction"):
        value = cfg.hardware.get(name)
        if value is not None and value not in (-1, 1):
            raise ValueError(f"hardware.{name}은 -1 또는 +1이어야 합니다.")
    gravity_sign = cfg.geometry.get("gravity_torque_sign")
    if gravity_sign is not None and gravity_sign not in (-1, 1):
        raise ValueError("gravity_torque_sign은 -1 또는 +1이어야 합니다.")
    inertia_reference = cfg.geometry.get("arm_inertia_reference")
    if inertia_reference is not None and inertia_reference not in ("about_com", "about_pivot"):
        raise ValueError("arm_inertia_reference는 about_com 또는 about_pivot이어야 합니다.")
    if cfg.execution_order is not None and cfg.execution_order not in ("grouped", "randomized", "blocked_randomized"):
        raise ValueError("execution_order는 grouped, randomized 또는 blocked_randomized여야 합니다.")
    for selected in (cfg.pilot.get("mechanical_configuration"), cfg.trajectories["delay_probe"].get("mechanical_configuration")):
        if selected is not None and selected not in ids:
            raise ValueError(f"mechanical configuration이 canonical six에 없습니다: {selected}")
    hardware_ranges = {
        "baudrate": (1, None), "motor_id": (0, 252), "expected_model_number": (1, None),
        "encoder_zero_raw": (0, 4095), "expected_homing_offset_raw": (-1044479, 1044479),
    }
    for name, (minimum, maximum) in hardware_ranges.items():
        value = cfg.hardware.get(name)
        if value is not None and (float(value) < minimum or (maximum is not None and float(value) > maximum)):
            raise ValueError(f"hardware.{name} 범위가 유효하지 않습니다: {value}")
    for name in ("target_generation_rate_hz", "bus_write_rate_hz", "state_read_rate_hz",
                 "delay_telemetry_target_rate_hz", "hardware_error_poll_rate_hz",
                 "severe_overrun_threshold_sec"):
        value = cfg.timing.get(name)
        if value is not None and float(value) <= 0:
            raise ValueError(f"timing.{name}은 양수여야 합니다.")
    for name in ("current_near_limit_fraction", "pwm_near_limit_fraction"):
        value = cfg.timing.get(name)
        if value is not None and not 0.0 < float(value) <= 1.0:
            raise ValueError(f"timing.{name}은 (0,1]이어야 합니다.")
    for name in ("position_p_gain", "position_i_gain", "position_d_gain", "feedforward_1st_gain", "feedforward_2nd_gain"):
        value = cfg.registers.get(name)
        if value is not None and not 0 <= int(value) <= 16383:
            raise ValueError(f"mode5_registers.{name}이 [0,16383] 밖입니다.")
    if cfg.registers.get("position_p_gain") is not None and int(cfg.registers["position_p_gain"]) <= 0:
        raise ValueError("position_p_gain은 실제 사용 양수값이어야 합니다.")
    watchdog = cfg.registers.get("bus_watchdog_raw")
    if watchdog is not None and not 1 <= int(watchdog) <= 127:
        raise ValueError("bus_watchdog_raw는 활성 범위 [1,127]이어야 합니다 (20 ms/count).")
    if cfg.registers.get("goal_current_raw") is not None and cfg.registers.get("expected_current_limit_raw") is not None:
        if not 0 < abs(int(cfg.registers["goal_current_raw"])) <= int(cfg.registers["expected_current_limit_raw"]):
            raise ValueError("Goal Current는 0보다 크고 Current Limit 이하여야 합니다.")
    if cfg.registers.get("goal_pwm_raw") is not None and cfg.registers.get("expected_pwm_limit_raw") is not None:
        if not 0 < abs(int(cfg.registers["goal_pwm_raw"])) <= int(cfg.registers["expected_pwm_limit_raw"]):
            raise ValueError("Goal PWM은 0보다 크고 PWM Limit 이하여야 합니다.")
    lo, hi = cfg.safety.get("software_position_min_rad"), cfg.safety.get("software_position_max_rad")
    if lo is not None and hi is not None and float(lo) >= float(hi):
        raise ValueError("software_position_min_rad는 max보다 작아야 합니다.")
    positive_safety = (
        "maximum_temperature_c", "minimum_input_voltage_v", "maximum_input_voltage_v",
        "maximum_abs_current_A", "maximum_abs_pwm_fraction", "maximum_abs_position_error_rad",
        "maximum_consecutive_overruns", "oscillation_velocity_limit_rad_s",
        "transition_duration_sec", "between_runs_sec",
    )
    for name in positive_safety:
        value = cfg.safety.get(name)
        if value is not None and float(value) <= 0:
            raise ValueError(f"safety.{name}은 양수여야 합니다.")
    pwm_fraction = cfg.safety.get("maximum_abs_pwm_fraction")
    if pwm_fraction is not None and float(pwm_fraction) > 1.0:
        raise ValueError("maximum_abs_pwm_fraction은 1.0 이하여야 합니다.")
    for name in ("arm_mass_kg", "arm_com_radius_m", "arm_inertia_kg_m2"):
        value = cfg.geometry.get(name)
        if value is not None and float(value) <= 0:
            raise ValueError(f"bench.geometry.{name}은 양수여야 합니다.")
    for name, value in cfg.geometry.get("arm_lengths_m", {}).items():
        if value is not None and float(value) <= 0:
            raise ValueError(f"bench.geometry.arm_lengths_m.{name}은 양수여야 합니다.")
    for name, load in cfg.loads.items():
        measured = load.get("measured_mass_kg")
        if measured is not None and float(measured) <= 0:
            raise ValueError(f"bench.loads.{name}.measured_mass_kg은 양수여야 합니다.")
