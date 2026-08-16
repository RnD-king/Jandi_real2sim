from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .bus import Mode5Bus, State
from .config import Mode5Campaign
from .trajectories import Sample, build, pilot_step


TELEMETRY_FIELDS = (
    "host_time_ns", "tx_start_ns", "tx_end_ns", "rx_end_ns", "cycle_index",
    "time_s", "phase", "acquisition_kind", "overrun_ns", "q_cmd_rad",
    "goal_position_tick", "q_present_rad", "present_position_tick",
    "dq_present_rad_s", "present_velocity_raw", "current_A_joint",
    "present_current_raw", "pwm_percent", "present_pwm_raw",
    "position_trajectory_rad", "position_trajectory_tick",
    "velocity_trajectory_rad_s", "velocity_trajectory_raw", "input_voltage_V",
    "temperature_C", "realtime_tick_ms", "moving", "moving_status",
    "hardware_error",
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_snapshot(
    cfg: Mode5Campaign, condition: str, trajectory: str
) -> dict[str, object]:
    return {
        "sources": {
            role: {"path": str(path), "sha256": _sha256(path)}
            for role, path in sorted(cfg.source_files.items())
        },
        "resolved": {
            "hardware": asdict(cfg.hardware),
            "timing": asdict(cfg.timing),
            "mode5_registers": asdict(cfg.registers),
            "bench": cfg.benches[condition].resolved_metadata(),
            "trajectory": {
                "name": trajectory,
                "parameters": cfg.trajectories[trajectory],
            },
            "safety": asdict(cfg.safety),
        },
    }


def _run_dir(cfg: Mode5Campaign, condition: str, trajectory: str, repeat: int, *, dry: bool, pilot: bool = False) -> Path:
    root = cfg.output_root if not dry else cfg.project_root / "data/plans/mode5"
    if pilot:
        return root / cfg.campaign_id / "pilot" / "no_load_step"
    return root / cfg.campaign_id / condition / trajectory / f"repeat_{repeat}"


def write_plan(cfg: Mode5Campaign, condition: str, trajectory: str, repeat: int, *, pilot: bool = False) -> Path:
    samples = pilot_step(cfg) if pilot else build(cfg, trajectory)
    run_dir = _run_dir(cfg, condition, trajectory, repeat, dry=True, pilot=pilot)
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "plan.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("cycle_index", "time_s", "phase", "goal_rad", "goal_tick"))
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "cycle_index": sample.cycle_index,
                "time_s": f"{sample.time_s:.9f}",
                "phase": sample.phase,
                "goal_rad": f"{sample.goal_rad:.9f}",
                "goal_tick": cfg.rad_to_tick(sample.goal_rad),
            })
    _json(run_dir / "metadata.json", {
        "data_kind": "dry_run_plan",
        "campaign_id": cfg.campaign_id,
        "condition": condition,
        "trajectory": trajectory,
        "repeat": repeat,
        "split_role": "validation" if repeat == 3 else "fit",
        "pilot": pilot,
        "sample_count": len(samples),
        "duration_s": len(samples) / cfg.timing.command_rate_hz,
        "config_snapshot": _config_snapshot(cfg, condition, trajectory),
    })
    return run_dir


class SafetyMonitor:
    def __init__(self, cfg: Mode5Campaign):
        self.cfg = cfg
        self.counts: dict[str, int] = {}

    def check(self, state: State, command_rad: float) -> None:
        direction = self.cfg.hardware.direction
        q = self.cfg.tick_to_rad(state.present_position_tick)
        values = {
            "temperature": (state.temperature_c >= self.cfg.safety.max_temperature_c, state.temperature_c),
            "voltage": (state.input_voltage_raw * 0.1 <= self.cfg.safety.min_input_voltage_v, state.input_voltage_raw * 0.1),
            "current": (abs(state.present_current_raw * 0.00336) >= self.cfg.safety.max_abs_current_a, abs(state.present_current_raw * 0.00336)),
            "pwm": (abs(state.present_pwm_raw * 0.113) >= self.cfg.safety.max_abs_pwm_percent, abs(state.present_pwm_raw * 0.113)),
            "position_error": (abs(command_rad - q) >= self.cfg.safety.max_abs_position_error_rad, abs(command_rad - q)),
        }
        for name, (violated, actual) in values.items():
            self.counts[name] = self.counts.get(name, 0) + 1 if violated else 0
            if self.counts[name] >= self.cfg.safety.consecutive_state_samples:
                raise RuntimeError(f"LIVE SAFETY {name}: actual={actual}")
        _ = direction


def _transition(start: float, target: float, duration: float, rate: int) -> list[Sample]:
    count = max(1, round(duration * rate))
    result = []
    for cycle in range(count + 1):
        ratio = cycle / count
        blend = 0.5 - 0.5 * math.cos(math.pi * ratio)
        result.append(Sample(cycle, cycle / rate, "unrecorded_transition", start + blend * (target - start)))
    return result


def _run_samples(
    bus: Mode5Bus,
    cfg: Mode5Campaign,
    samples: Iterable[Sample],
    telemetry: csv.DictWriter | None,
    events: csv.DictWriter | None,
    monitor: SafetyMonitor,
) -> int:
    period_ns = round(1e9 / cfg.timing.command_rate_hz)
    start_ns = time.monotonic_ns()
    count = 0
    for sample in samples:
        deadline = start_ns + sample.cycle_index * period_ns
        remaining = deadline - time.monotonic_ns()
        if remaining > 0:
            time.sleep(remaining / 1e9)
        tx_start = time.monotonic_ns()
        bus.write_goal_rad(sample.goal_rad)
        tx_end = time.monotonic_ns()
        if events is not None:
            events.writerow({
                "sequence": sample.cycle_index,
                "scheduled_time_s": f"{sample.time_s:.9f}",
                "phase": sample.phase,
                "goal_rad": f"{sample.goal_rad:.9f}",
                "goal_tick": cfg.rad_to_tick(sample.goal_rad),
                "tx_start_ns": tx_start,
                "tx_end_ns": tx_end,
            })
        slot = sample.cycle_index % cfg.timing.command_rate_hz
        state: State | None = None
        error: int | None = None
        if slot < cfg.timing.state_read_rate_hz:
            state = bus.read_state()
            monitor.check(state, sample.goal_rad)
            kind = "state"
        else:
            error = bus.read_hardware_error()
            kind = "hardware_error"
            if error:
                raise RuntimeError(f"Hardware Error Status={error}")
        rx_end = time.monotonic_ns()
        if telemetry is not None:
            row = {field: "" for field in TELEMETRY_FIELDS}
            row.update({
                "host_time_ns": rx_end,
                "tx_start_ns": tx_start,
                "tx_end_ns": tx_end,
                "rx_end_ns": rx_end,
                "cycle_index": sample.cycle_index,
                "time_s": f"{sample.time_s:.9f}",
                "phase": sample.phase,
                "acquisition_kind": kind,
                "overrun_ns": max(0, rx_end - (deadline + period_ns)),
                "q_cmd_rad": f"{sample.goal_rad:.9f}",
                "goal_position_tick": cfg.rad_to_tick(sample.goal_rad),
                "hardware_error": "" if error is None else error,
            })
            if state is not None:
                velocity_unit = 0.229 * 2 * math.pi / 60
                direction = cfg.hardware.direction
                row.update({
                    "q_present_rad": f"{cfg.tick_to_rad(state.present_position_tick):.9f}",
                    "present_position_tick": state.present_position_tick,
                    "dq_present_rad_s": f"{direction * state.present_velocity_raw * velocity_unit:.9f}",
                    "present_velocity_raw": state.present_velocity_raw,
                    "current_A_joint": f"{direction * state.present_current_raw * 0.00336:.9f}",
                    "present_current_raw": state.present_current_raw,
                    "pwm_percent": f"{direction * state.present_pwm_raw * 0.113:.6f}",
                    "present_pwm_raw": state.present_pwm_raw,
                    "position_trajectory_rad": f"{cfg.tick_to_rad(state.position_trajectory_tick):.9f}",
                    "position_trajectory_tick": state.position_trajectory_tick,
                    "velocity_trajectory_rad_s": f"{direction * state.velocity_trajectory_raw * velocity_unit:.9f}",
                    "velocity_trajectory_raw": state.velocity_trajectory_raw,
                    "input_voltage_V": f"{state.input_voltage_raw * 0.1:.3f}",
                    "temperature_C": state.temperature_c,
                    "realtime_tick_ms": state.realtime_tick_ms,
                    "moving": state.moving,
                    "moving_status": state.moving_status,
                })
            telemetry.writerow(row)
        count += 1
    return count


def collect(cfg: Mode5Campaign, condition: str, trajectory: str, repeat: int, *, pilot: bool = False) -> Path:
    unresolved = cfg.unresolved_for_execution(
        condition=condition,
        require_pilot_approval=not pilot,
    )
    if unresolved:
        raise RuntimeError("실기체 실행 전 미확정 항목: " + ", ".join(unresolved))
    samples = pilot_step(cfg) if pilot else build(cfg, trajectory)
    run_dir = _run_dir(cfg, condition, trajectory, repeat, dry=False, pilot=pilot)
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = run_dir / "metadata.json"
    metadata: dict[str, object] = {
        "valid_flag": False,
        "campaign_id": cfg.campaign_id,
        "condition": condition,
        "trajectory": trajectory,
        "repeat": repeat,
        "split_role": "validation" if repeat == 3 else "fit",
        "pilot": pilot,
        "config_snapshot": _config_snapshot(cfg, condition, trajectory),
        "bench": cfg.benches[condition].resolved_metadata(),
        "started_at": datetime.now().astimezone().isoformat(),
    }
    try:
        with Mode5Bus(cfg) as bus:
            metadata["model_number"] = bus.ping()
            actual = bus.configure_and_verify()
            metadata["register_readback"] = actual
            initial = bus.read_state()
            current_rad = cfg.tick_to_rad(initial.present_position_tick)
            cfg.validate_motion(current_rad)
            bus.write_goal_rad(current_rad)
            bus.torque(True)
            try:
                monitor = SafetyMonitor(cfg)
                transition = _transition(current_rad, cfg.hardware.center_rad, cfg.timing.transition_sec, cfg.timing.command_rate_hz)
                _run_samples(bus, cfg, transition, None, None, monitor)
                with (run_dir / "telemetry.csv").open("x", newline="") as telemetry_stream, (run_dir / "command_events.csv").open("x", newline="") as event_stream:
                    telemetry = csv.DictWriter(telemetry_stream, fieldnames=TELEMETRY_FIELDS)
                    telemetry.writeheader()
                    event_fields = ("sequence", "scheduled_time_s", "phase", "goal_rad", "goal_tick", "tx_start_ns", "tx_end_ns")
                    events = csv.DictWriter(event_stream, fieldnames=event_fields)
                    events.writeheader()
                    metadata["sample_count"] = _run_samples(bus, cfg, samples, telemetry, events, monitor)
            finally:
                bus.torque(False)
    except BaseException as exc:
        metadata["invalid_reason"] = repr(exc)
        _json(metadata_path, metadata)
        raise
    metadata["valid_flag"] = True
    metadata["invalid_reason"] = ""
    metadata["finished_at"] = datetime.now().astimezone().isoformat()
    _json(metadata_path, metadata)
    return run_dir
