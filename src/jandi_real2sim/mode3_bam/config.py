"""Single source of truth for the Mode-3 BAM campaign."""

from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAMPAIGN = PROJECT_ROOT / "configs/mode3_bam/campaign.yaml"
MASS_KEYS = ("mass1", "mass2", "mass3")
DISTANCE_KEYS = ("distance1", "distance2")
LOADED_TRAJECTORIES = ("sin_time_square", "sin_sin", "up_and_down", "lift_and_drop")
NO_LOAD_TRAJECTORIES = ("delay_probe", "backlash_probe")


def trajectory_profile_matches(cfg: "Campaign", trajectory: str,
                               metadata: dict[str, Any]) -> bool:
    """Reject valid attempts collected with an obsolete trajectory profile."""
    expected = int(cfg.trajectories[trajectory].get("profile_version", 1))
    actual = int(metadata.get("trajectory_profile_version", 1))
    return actual == expected


@dataclass(frozen=True)
class Condition:
    id: str
    mass_key: str | None
    distance_key: str | None
    disk_count: int
    mass_kg: float
    distance_m: float
    intrinsic_inertia_kg_m2: float

    @property
    def loaded(self) -> bool:
        return self.mass_key is not None


@dataclass(frozen=True)
class Campaign:
    source: Path
    project_root: Path
    campaign: dict[str, Any]
    hardware: dict[str, Any]
    timing: dict[str, Any]
    registers: dict[str, Any]
    bench: dict[str, Any]
    trajectories: dict[str, Any]
    safety: dict[str, Any]
    confirmations: dict[str, Any]
    fit: dict[str, Any]
    repetitions: tuple[int, ...]
    fit_repetitions: tuple[int, ...]
    validation_repetitions: tuple[int, ...]
    source_files: dict[str, Path]

    @property
    def campaign_id(self) -> str | None:
        return self.campaign.get("id")

    @property
    def output_root(self) -> Path:
        return (self.project_root / str(self.campaign["output_root"])).resolve()

    @property
    def results_root(self) -> Path:
        return (self.project_root / str(self.campaign["results_root"])).resolve()

    @property
    def command_rate_hz(self) -> float:
        return float(self.timing["command_rate_hz"])

    @property
    def conditions(self) -> tuple[Condition, ...]:
        result = [Condition("no_load", None, None, 0, 0.0, 0.0, 0.0)]
        load = self.bench["load"]
        disk_mass = float(load["disk_mass_kg"])
        disk_radius = 0.5 * float(load["disk_diameter_m"])
        disk_inertia = 0.5 * disk_mass * disk_radius * disk_radius
        fastener_mass = float(load["fastener_mass_kg"])
        for mass_key in MASS_KEYS:
            disk_count = int(load["disk_counts"][mass_key])
            mass = disk_count * disk_mass + fastener_mass
            intrinsic_inertia = disk_count * disk_inertia
            for distance_key in DISTANCE_KEYS:
                distance = self.bench["axis_to_load_com_m"].get(distance_key)
                result.append(Condition(
                    f"{mass_key}_{distance_key}", mass_key, distance_key,
                    disk_count, mass,
                    0.0 if distance is None else float(distance),
                    intrinsic_inertia,
                ))
        return tuple(result)

    def condition(self, condition_id: str) -> Condition:
        try:
            return next(item for item in self.conditions if item.id == condition_id)
        except StopIteration as exc:
            raise KeyError(f"unknown condition: {condition_id}") from exc

    def raw_to_rad(self, raw: int) -> float:
        return int(self.hardware["direction"]) * (raw - int(self.hardware["encoder_zero_raw"])) * 2.0 * 3.141592653589793 / 4096.0

    def rad_to_raw(self, rad: float) -> int:
        return round(int(self.hardware["encoder_zero_raw"]) + int(self.hardware["direction"]) * rad * 4096.0 / (2.0 * 3.141592653589793))


def _read(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"YAML mapping이 아닙니다: {path}")
    return value


def load_campaign(path: str | Path = DEFAULT_CAMPAIGN, *, require_hardware: bool = False,
                  require_bench: bool = False) -> Campaign:
    source = Path(path).resolve()
    root = source.parents[2]
    raw = _read(source)
    if raw.get("schema_version") != 1:
        raise ValueError("Mode-3 BAM campaign schema_version은 1이어야 합니다.")
    components = raw["components"]
    files = {name: (source.parent / relative).resolve() for name, relative in components.items()}
    docs = {name: _read(file) for name, file in files.items()}
    cfg = Campaign(
        source, root, dict(raw["campaign"]), dict(docs["hardware"]["hardware"]),
        dict(docs["hardware"]["timing"]), dict(docs["controller"]["mode3_registers"]),
        dict(docs["bench"]["bench"]), dict(docs["trajectories"]["trajectories"]),
        dict(docs["safety"]["safety"]), dict(docs["safety"]["confirmations"]),
        dict(docs["fit"]["fit"]), tuple(map(int, raw["repetitions"])),
        tuple(map(int, raw["fit_repetitions"])), tuple(map(int, raw["validation_repetitions"])),
        {"campaign": source, **files},
    )
    _validate(cfg, require_hardware=require_hardware, require_bench=require_bench)
    return cfg


def _validate(cfg: Campaign, *, require_hardware: bool, require_bench: bool) -> None:
    if cfg.registers.get("operating_mode") != 3:
        raise ValueError("Mode-3 BAM campaign은 operating_mode=3만 허용합니다.")
    if int(cfg.registers.get("position_p_gain") or 0) != 850:
        raise ValueError("현재 canonical 실험은 position_p_gain=850으로 고정합니다.")
    if cfg.repetitions != (1, 2, 3) or cfg.fit_repetitions != (1, 2) or cfg.validation_repetitions != (3,):
        raise ValueError("repeat 계약은 fit=1,2 / validation=3입니다.")
    if len(MASS_KEYS) * len(DISTANCE_KEYS) + 1 != 7:
        raise AssertionError("7-condition matrix contract broken")
    if float(cfg.timing.get("hardware_error_poll_rate_hz") or 0) <= 0:
        raise ValueError("hardware_error_poll_rate_hz는 양수여야 합니다.")
    required_hw = ("serial_device", "baudrate", "motor_id", "expected_model_number",
                   "encoder_zero_raw", "expected_homing_offset_raw", "direction",
                   "current_direction", "pwm_direction")
    if require_hardware:
        missing = [name for name in required_hw if cfg.hardware.get(name) is None]
        missing += [name for name in ("drive_mode", "position_d_gain", "expected_pwm_limit_raw",
                                      "expected_current_limit_raw", "bus_watchdog_raw")
                    if cfg.registers.get(name) is None]
        missing += [name for name, value in cfg.safety.items() if value is None]
        if missing:
            raise ValueError(f"실기체 실행 전 필수 설정이 비었습니다: {', '.join(missing)}")
    if require_bench:
        missing = [f"axis_to_load_com_m.{key}" for key in DISTANCE_KEYS
                   if cfg.bench["axis_to_load_com_m"].get(key) is None]
        missing += [f"load.{name}" for name in ("disk_mass_kg", "disk_diameter_m", "fastener_mass_kg")
                    if cfg.bench["load"].get(name) is None]
        missing += [f"load.disk_counts.{key}" for key in MASS_KEYS
                    if cfg.bench["load"]["disk_counts"].get(key) is None]
        missing += [f"arm.{name}" for name in ("mass_kg", "com_radius_m", "inertia_about_com_kg_m2",
                                                "inertia_about_pivot_kg_m2")
                    if cfg.bench["arm"].get(name) is None]
        if cfg.bench["no_load_hardware"].get("horn_mass_kg") is None:
            missing.append("no_load_hardware.horn_mass_kg")
        missing += [f"loaded_mounting_hardware.{name}"
                    for name in ("horn_mass_kg", "horn_fastener_mass_kg")
                    if cfg.bench["loaded_mounting_hardware"].get(name) is None]
        if cfg.bench.get("gravity_torque_sign") is None:
            missing.append("gravity_torque_sign")
        if missing:
            raise ValueError(f"부하 실험 전 bench 실측값이 비었습니다: {', '.join(missing)}")
    values = [cfg.bench["load"].get(name) for name in ("disk_mass_kg", "disk_diameter_m", "fastener_mass_kg")]
    values += [v for v in cfg.bench["axis_to_load_com_m"].values() if v is not None]
    values += [cfg.bench["arm"].get(name) for name in ("mass_kg", "com_radius_m",
                                                       "inertia_about_com_kg_m2",
                                                       "inertia_about_pivot_kg_m2")]
    values += [cfg.bench["no_load_hardware"].get("horn_mass_kg")]
    values += [cfg.bench["loaded_mounting_hardware"].get(name)
               for name in ("horn_mass_kg", "horn_fastener_mass_kg")]
    values = [value for value in values if value is not None]
    if any(float(value) <= 0 for value in values):
        raise ValueError("질량, 치수, 관성 및 축-질량중심 거리는 양수여야 합니다.")
    counts = [cfg.bench["load"]["disk_counts"].get(key) for key in MASS_KEYS]
    if any(value is not None and (int(value) != value or int(value) <= 0) for value in counts):
        raise ValueError("load.disk_counts는 양의 정수여야 합니다.")


EDITABLE = {
    "campaign": ("campaign", "campaign", "id"),
    **{key: ("bench", "axis_to_load_com_m", key) for key in DISTANCE_KEYS},
    "disk_mass_kg": ("bench", "load", "disk_mass_kg"),
    "disk_diameter_m": ("bench", "load", "disk_diameter_m"),
    "load_fastener_mass_kg": ("bench", "load", "fastener_mass_kg"),
    "no_load_horn_mass_kg": ("bench", "no_load_hardware", "horn_mass_kg"),
    "loaded_horn_mass_kg": ("bench", "loaded_mounting_hardware", "horn_mass_kg"),
    "loaded_horn_fastener_mass_kg": ("bench", "loaded_mounting_hardware", "horn_fastener_mass_kg"),
    "arm_mass_kg": ("bench", "arm", "mass_kg"),
    "arm_com_radius_m": ("bench", "arm", "com_radius_m"),
    "arm_inertia_about_com_kg_m2": ("bench", "arm", "inertia_about_com_kg_m2"),
    "arm_inertia_kg_m2": ("bench", "arm", "inertia_about_pivot_kg_m2"),
}


def update_value(campaign_path: Path, key: str, value: Any) -> None:
    if key not in EDITABLE:
        raise KeyError(f"GUI editable field가 아닙니다: {key}")
    cfg = load_campaign(campaign_path)
    role, *parts = EDITABLE[key]
    source = cfg.source if role == "campaign" else cfg.source_files[role]
    document = _read(source)
    target = document
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    original = source.read_text()
    fd, name = tempfile.mkstemp(prefix=source.name + ".candidate.", dir=source.parent)
    os.close(fd)
    candidate = Path(name)
    try:
        candidate.write_text(yaml.safe_dump(document, sort_keys=False))
        candidate.replace(source)
        try:
            load_campaign(campaign_path)
        except BaseException:
            source.write_text(original)
            raise
    finally:
        candidate.unlink(missing_ok=True)
