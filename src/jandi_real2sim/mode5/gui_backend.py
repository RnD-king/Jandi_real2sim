"""Hardware-free GUI services built only from canonical configuration/backend APIs."""

from __future__ import annotations

import copy
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .canonical_acquisition import run_directory
from .canonical_attempts import inspect_logical_run
from .canonical_config import CanonicalCampaign, load_canonical_campaign
from .canonical_trajectories import (
    Sample, build_delay, build_dynamic, build_static, dynamic_run_specs, static_run_specs,
)


@dataclass(frozen=True)
class Preview:
    samples: tuple[Sample, ...]
    duration_sec: float
    minimum_rad: float
    maximum_rad: float
    maximum_discrete_speed_rad_s: float


def build_preview(cfg: CanonicalCampaign, mode: str, trajectory: str = "",
                  approach: str = "approach_positive") -> Preview:
    if mode == "static":
        samples = build_static(cfg, approach)
    elif mode == "delay":
        samples = build_delay(cfg)
    elif mode == "dynamic":
        samples = build_dynamic(cfg, trajectory)
    else:
        raise ValueError(f"canonical preview mode가 아닙니다: {mode}")
    values = [sample.goal_position_rad for sample in samples]
    rate = cfg.command_rate_hz
    speed = max((abs(b - a) * rate for a, b in zip(values, values[1:])), default=0.0)
    return Preview(tuple(samples), samples[-1].scheduled_time_sec if samples else 0.0,
                   min(values), max(values), speed)


def progress_rows(cfg: CanonicalCampaign, mode: str) -> list[dict[str, Any]]:
    specs = static_run_specs(cfg) if mode == "static" else dynamic_run_specs(cfg)
    rows = []
    for spec in specs:
        logical = run_directory(cfg, spec.relative_directory)
        status = inspect_logical_run(logical)
        rows.append({
            "relative_directory": spec.relative_directory,
            "mechanical_configuration": spec.mechanical_configuration,
            "trajectory": spec.trajectory,
            "approach_direction": getattr(spec, "approach_direction", None),
            "repeat": spec.repeat,
            "state": status.state,
            "attempt_count": len(status.attempts),
            "invalid_attempt_count": len(status.invalid_attempts) + len(status.incomplete_attempts),
            "valid_attempts": [path.name for path in status.valid_attempts],
            "selected_attempt": status.selected_attempt.name if status.selected_attempt else None,
        })
    return rows


EDITABLE_FIELDS = {
    "campaign": {"campaign.id", "holdout_configuration", "execution_order", "randomization_seed"},
    "bench.loads": {"loads.m250.measured_mass_kg", "loads.m500.measured_mass_kg", "loads.m750.measured_mass_kg"},
    "bench.geometry": {"arm_mass_kg", "arm_com_radius_m", "arm_inertia_kg_m2", "arm_inertia_reference",
                       "gravity_torque_sign", "physical_fixture_axis_description",
                       "coordinate.positive_direction", "coordinate.current_positive_direction",
                       "coordinate.gravity_torque_positive_direction", "coordinate.mujoco_positive_direction"},
    "controller": {
        "mode5_registers.drive_mode",
        "mode5_registers.position_p_gain", "mode5_registers.position_d_gain",
        "mode5_registers.goal_current_raw", "mode5_registers.expected_current_limit_raw",
        "mode5_registers.goal_pwm_raw", "mode5_registers.expected_pwm_limit_raw",
        "mode5_registers.bus_watchdog_raw",
    },
    "hardware": {
        "hardware.serial_device", "hardware.baudrate", "hardware.motor_id",
        "hardware.expected_model_number", "hardware.encoder_zero_raw", "hardware.direction",
        "hardware.expected_homing_offset_raw",
        "hardware.current_direction", "hardware.pwm_direction",
        "timing.delay_telemetry_target_rate_hz", "timing.hardware_error_poll_rate_hz",
        "timing.severe_overrun_threshold_sec", "timing.current_near_limit_fraction",
        "timing.pwm_near_limit_fraction",
    },
    "safety": {
        "safety.software_position_min_rad", "safety.software_position_max_rad",
        "safety.maximum_temperature_c", "safety.minimum_input_voltage_v",
        "safety.maximum_input_voltage_v", "safety.maximum_abs_current_A",
        "safety.maximum_abs_pwm_fraction", "safety.maximum_abs_position_error_rad",
        "safety.maximum_consecutive_overruns", "safety.oscillation_velocity_limit_rad_s",
        "safety.transition_duration_sec", "safety.between_runs_sec",
        "safety.warmup_procedure", "approval.pilot_passed", "approval.pilot_run_reference",
        "approval.pilot_approved_at", "approval.warmup_acknowledged_at",
        "pilot.mechanical_configuration", "pilot.center_rad", "pilot.amplitude_rad", "pilot.hold_sec",
    },
}
EDITABLE_FIELDS["fit"] = {
    "physics_timestep_sec", "static_bootstrap.repeat_count", "static_bootstrap.random_seed",
    "static_bootstrap.condition_number_warning_threshold", "stage_d.parameter_bounds",
    "stage_d.initial_parameters", "stage_d.independent_seeds", "stage_d.max_function_evaluations",
    "stage_d.evaluation_stride", "loss.position_scale_rad", "loss.velocity_scale_rad_s",
    "loss.current_scale_A", "loss.weights", "stage_e.enabled", "stage_e.uncertainty_bounds",
}

EDITABLE_FIELDS["trajectory.static_calibration"] = {
    "parameters.approach_offset_rad", "parameters.approach_duration_sec",
    "parameters.inter_point_transfer_duration_sec", "parameters.fixed_settling_hold_sec",
    "parameters.minimum_settling_sec", "parameters.averaging_window_sec",
    "parameters.maximum_command_speed_rad_s", "parameters.maximum_settled_abs_velocity_rad_s",
    "parameters.maximum_settled_position_std_rad", "parameters.maximum_settled_current_std_A",
}
EDITABLE_FIELDS["trajectory.delay_probe"] = {
    "parameters.mechanical_configuration", "parameters.center_rad", "parameters.step_amplitudes_rad",
    "parameters.hold_sec", "parameters.repeats", "parameters.pre_event_baseline_sec",
    "parameters.onset_current_threshold_A", "parameters.response_search_sec",
}
EDITABLE_FIELDS["trajectory.accelerated_oscillation"] = {
    "parameters.center_rad", "parameters.amplitude_rad", "parameters.start_frequency_hz",
    "parameters.end_frequency_hz", "parameters.duration_sec", "parameters.center_hold_sec",
    "parameters.transition_duration_sec", "parameters.maximum_command_speed_rad_s",
}
EDITABLE_FIELDS["trajectory.slow_plus_highfreq"] = {
    "parameters.center_rad", "parameters.slow_amplitude_rad", "parameters.slow_frequency_hz",
    "parameters.high_frequency_amplitude_rad", "parameters.high_frequency_hz",
    "parameters.duration_sec", "parameters.center_hold_sec", "parameters.transition_duration_sec",
    "parameters.maximum_command_speed_rad_s",
}
EDITABLE_FIELDS["trajectory.slowly_raise_lower"] = {
    "parameters.center_rad", "parameters.lower_rad", "parameters.upper_rad", "parameters.speed_rad_s",
    "parameters.cycles", "parameters.endpoint_hold_sec", "parameters.center_hold_sec",
    "parameters.transition_duration_sec", "parameters.maximum_command_speed_rad_s",
}


def _set_nested(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    target = mapping
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"mapping 경로가 아닙니다: {dotted}")
        target = child
    target[parts[-1]] = value


def validated_config_update(campaign_path: Path, role: str, dotted: str, value: Any) -> None:
    """Atomically save one whitelisted field only if the whole campaign still validates."""
    cfg = load_canonical_campaign(campaign_path)
    freeze = cfg.output_root / str(cfg.campaign_id) / "campaign_freeze.json" if cfg.campaign_id else None
    if freeze is not None and freeze.exists():
        raise PermissionError("campaign freeze 이후 critical config는 수정할 수 없습니다. NEW CAMPAIGN을 만드십시오.")
    if dotted not in EDITABLE_FIELDS.get(role, set()):
        raise ValueError(f"GUI에서 수정할 수 없는 canonical field입니다: {role}:{dotted}")
    source = (cfg.project_root / "configs/mode5/fit.yaml") if role == "fit" else cfg.source_files[role]
    original = source.read_text()
    document = yaml.safe_load(original)
    updated = copy.deepcopy(document)
    _set_nested(updated, dotted, value)
    fd, temporary_name = tempfile.mkstemp(prefix=source.name + ".candidate.", dir=source.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    fd, backup_name = tempfile.mkstemp(prefix=source.name + ".backup.", dir=source.parent)
    os.close(fd)
    backup = Path(backup_name)
    try:
        temporary.write_text(yaml.safe_dump(updated, sort_keys=False))
        backup.write_text(original)
        mode = source.stat().st_mode
        temporary.chmod(mode)
        backup.chmod(mode)
        temporary.replace(source)
        try:
            load_canonical_campaign(campaign_path)
        except BaseException:
            backup.replace(source)
            raise
    finally:
        temporary.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)


def require_physical_confirmation(value: dict[str, Any] | None) -> dict[str, Any]:
    if not value or not value.get("confirmed_at") or not value.get("mechanical_configuration"):
        raise PermissionError("PHYSICAL SETUP CONFIRMATION이 없어서 실기체 run을 차단했습니다.")
    return value


def completed_run_summary(path: Path, cfg: CanonicalCampaign) -> dict[str, Any]:
    metadata = json.loads((path / "metadata.json").read_text())
    with (path / "telemetry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary: dict[str, Any] = {
        "valid_flag": bool(metadata.get("valid_flag")), "invalid_reason": metadata.get("invalid_reason", ""),
        "temperature_start_C": metadata.get("temperature_start_C"), "temperature_end_C": metadata.get("temperature_end_C"),
        "measured_target_update_rate_hz": metadata.get("measured_target_update_rate_hz"),
        "measured_bus_write_rate_hz": metadata.get("measured_bus_write_rate_hz"),
        "measured_state_rate_hz": metadata.get("measured_state_rate_hz"),
        "goal_readback_mismatch_count": sum(int(row.get("goal_readback_mismatch", 0)) for row in rows),
        "timing_invalid_sample_count": sum(int(row.get("timing_invalid", 0)) for row in rows),
        "current_saturated_sample_count": sum(int(row["current_saturated"]) for row in rows),
        "pwm_saturated_sample_count": sum(int(row["pwm_saturated"]) for row in rows),
    }
    if metadata.get("experiment") == "static":
        phases = sorted({row["phase"] for row in rows if row["phase"].endswith("_averaging")})
        accepted = 0
        for phase in phases:
            selected = [row for row in rows if row["phase"] == phase]
            velocity = [abs(float(row["present_velocity_rad_s"])) for row in selected]
            position = [float(row["present_position_rad"]) for row in selected]
            current = [float(row["present_current_A"]) for row in selected]
            import statistics
            good = bool(selected) and not any(int(row["current_saturated"]) or int(row["pwm_saturated"]) for row in selected)
            good = good and statistics.median(velocity) <= float(cfg.trajectories["static_calibration"]["maximum_settled_abs_velocity_rad_s"])
            good = good and statistics.pstdev(position) <= float(cfg.trajectories["static_calibration"]["maximum_settled_position_std_rad"])
            good = good and statistics.pstdev(current) <= float(cfg.trajectories["static_calibration"]["maximum_settled_current_std_A"])
            accepted += int(good)
        summary["accepted_static_plateau_count"] = accepted
        summary["rejected_static_plateau_count"] = len(phases) - accepted
    return summary
