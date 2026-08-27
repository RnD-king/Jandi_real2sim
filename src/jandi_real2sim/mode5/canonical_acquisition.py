"""Immutable raw-data acquisition for the README-v3 Mode-5 bench."""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from .canonical_bus import CanonicalMode5Bus, State
from .canonical_attempts import allocate_attempt
from .canonical_config import CanonicalCampaign
from .canonical_trajectories import Sample
from .canonical_trajectories import command_events
from .spec import REALTIME_TICK_MODULUS


CURRENT_A_PER_RAW = 0.00336
PWM_FRACTION_PER_RAW = 0.00113
VELOCITY_RAD_S_PER_RAW = 0.229 * 2.0 * math.pi / 60.0

TELEMETRY_FIELDS = (
    "sample_index", "host_time_ns", "host_time_sec", "command_seq", "command_event",
    "command_tx_before_ns", "command_tx_after_ns", "scheduled_time_sec", "phase",
    "goal_position_raw", "goal_position_rad", "goal_position_readback_raw",
    "goal_position_readback_rad", "realtime_tick_raw",
    "realtime_tick_unwrapped_ms", "present_position_raw", "present_position_rad",
    "present_velocity_raw", "present_velocity_rad_s", "present_current_raw",
    "present_current_A", "present_pwm_raw", "present_pwm_fraction",
    "velocity_trajectory_raw", "velocity_trajectory_rad_s",
    "position_trajectory_raw", "position_trajectory_rad", "input_voltage_raw",
    "input_voltage_V", "temperature_C", "moving", "moving_status",
    "current_saturated", "pwm_saturated", "cycle_overrun_ns", "validity_tag",
)


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class TickUnwrapper:
    def __init__(self) -> None:
        self.previous: int | None = None
        self.offset = 0

    def update(self, raw: int) -> int:
        if not 0 <= raw < REALTIME_TICK_MODULUS:
            raise ValueError(f"Realtime Tick raw={raw}가 [0,{REALTIME_TICK_MODULUS - 1}] 밖입니다.")
        # A real wrap is a large high-to-low jump.  Small backward changes are
        # not silently promoted to another epoch; they remain visible as a
        # non-monotonic sample for validity diagnostics.
        if self.previous is not None and raw < self.previous:
            backward = self.previous - raw
            if backward > REALTIME_TICK_MODULUS // 2:
                self.offset += REALTIME_TICK_MODULUS
        self.previous = raw
        return self.offset + raw


class OperatorAbort(RuntimeError):
    """Raised at an acquisition-cycle boundary after a GUI/CLI abort request."""


@dataclass(frozen=True)
class AcquisitionStats:
    sample_count: int
    command_tx_after_ns: tuple[int, ...]
    state_rx_ns: tuple[int, ...]
    overruns_ns: tuple[int, ...]


def timing_statistics(stats: AcquisitionStats) -> dict[str, float | int]:
    def interval(values: tuple[int, ...], prefix: str) -> dict[str, float]:
        if len(values) < 2:
            return {f"measured_{prefix}_rate_hz": 0.0, f"{prefix}_interval_mean_ms": 0.0,
                    f"{prefix}_interval_std_ms": 0.0, f"{prefix}_interval_max_ms": 0.0}
        delta = [(right - left) * 1e-6 for left, right in zip(values, values[1:])]
        mean_ms = sum(delta) / len(delta)
        variance = sum((value - mean_ms) ** 2 for value in delta) / len(delta)
        return {f"measured_{prefix}_rate_hz": 1000.0 / mean_ms, f"{prefix}_interval_mean_ms": mean_ms,
                f"{prefix}_interval_std_ms": math.sqrt(variance), f"{prefix}_interval_max_ms": max(delta)}

    result: dict[str, float | int] = {"sample_count": stats.sample_count}
    result.update(interval(stats.command_tx_after_ns, "command"))
    result.update(interval(stats.state_rx_ns, "state"))
    positive = [value for value in stats.overruns_ns if value > 0]
    result["deadline_overrun_count"] = len(positive)
    result["deadline_overrun_max_ns"] = max(positive, default=0)
    if len(stats.state_rx_ns) >= 2:
        intervals = [right - left for left, right in zip(stats.state_rx_ns, stats.state_rx_ns[1:])]
        result["measured_sampling_resolution_s"] = float(sorted(intervals)[len(intervals) // 2] * 1e-9)
    else:
        result["measured_sampling_resolution_s"] = 0.0
    return result


class SafetyMonitor:
    def __init__(self, cfg: CanonicalCampaign):
        self.cfg = cfg
        self.overruns = 0

    def check(self, state: State, goal_rad: float, overrun_ns: int) -> None:
        q = self.cfg.raw_to_rad(state.present_position_raw)
        direction = int(self.cfg.hardware["direction"])
        velocity = direction * state.present_velocity_raw * VELOCITY_RAD_S_PER_RAW
        current = int(self.cfg.hardware["current_direction"]) * state.present_current_raw * CURRENT_A_PER_RAW
        pwm = int(self.cfg.hardware["pwm_direction"]) * state.present_pwm_raw * PWM_FRACTION_PER_RAW
        voltage = state.input_voltage_raw * 0.1
        checks = {
            "position": not float(self.cfg.safety["software_position_min_rad"]) <= q <= float(self.cfg.safety["software_position_max_rad"]),
            "position_error": abs(goal_rad - q) > float(self.cfg.safety["maximum_abs_position_error_rad"]),
            "temperature": state.temperature_c >= float(self.cfg.safety["maximum_temperature_c"]),
            "voltage": not float(self.cfg.safety["minimum_input_voltage_v"]) <= voltage <= float(self.cfg.safety["maximum_input_voltage_v"]),
            "current": abs(current) >= float(self.cfg.safety["maximum_abs_current_A"]),
            "pwm": abs(pwm) >= float(self.cfg.safety["maximum_abs_pwm_fraction"]),
            "velocity": abs(velocity) >= float(self.cfg.safety["oscillation_velocity_limit_rad_s"]),
        }
        violated = [name for name, value in checks.items() if value]
        if violated:
            raise RuntimeError(f"LIVE SAFETY {','.join(violated)} q={q:.6f}, I={current:.4f}, pwm={pwm:.4f}")
        self.overruns = self.overruns + 1 if overrun_ns > 0 else 0
        if self.overruns >= int(self.cfg.safety["maximum_consecutive_overruns"]):
            raise RuntimeError(f"LIVE SAFETY repeated_overrun={self.overruns}")


def _transition(start: float, target: float, cfg: CanonicalCampaign) -> list[Sample]:
    duration = float(cfg.safety["transition_duration_sec"])
    count = max(2, round(duration * cfg.command_rate_hz))
    result = []
    for index in range(count):
        ratio = index / (count - 1)
        blend = 0.5 - 0.5 * math.cos(math.pi * ratio)
        result.append(Sample(index, index / cfg.command_rate_hz, "unrecorded_transition", start + blend * (target - start)))
    return result


def _row(cfg: CanonicalCampaign, sample: Sample, state: State, tx0: int, tx1: int, rx: int,
         overrun: int, tick_ms: int, command_seq: int, command_event: bool) -> dict[str, object]:
    direction = int(cfg.hardware["direction"])
    current_direction = int(cfg.hardware["current_direction"])
    pwm_direction = int(cfg.hardware["pwm_direction"])
    current_cap_raw = min(abs(int(cfg.registers["goal_current_raw"])), int(cfg.registers["expected_current_limit_raw"]))
    pwm_cap_raw = min(abs(int(cfg.registers["goal_pwm_raw"])), int(cfg.registers["expected_pwm_limit_raw"]))
    return {
        "sample_index": sample.sample_index, "host_time_ns": rx, "host_time_sec": f"{rx * 1e-9:.9f}",
        "command_seq": command_seq, "command_event": int(command_event),
        "command_tx_before_ns": tx0, "command_tx_after_ns": tx1,
        "scheduled_time_sec": f"{sample.scheduled_time_sec:.9f}", "phase": sample.phase,
        "goal_position_raw": cfg.rad_to_raw(sample.goal_position_rad), "goal_position_rad": f"{sample.goal_position_rad:.9f}",
        "goal_position_readback_raw": state.goal_position_raw,
        "goal_position_readback_rad": f"{cfg.raw_to_rad(state.goal_position_raw):.9f}",
        "realtime_tick_raw": state.realtime_tick_raw, "realtime_tick_unwrapped_ms": tick_ms,
        "present_position_raw": state.present_position_raw, "present_position_rad": f"{cfg.raw_to_rad(state.present_position_raw):.9f}",
        "present_velocity_raw": state.present_velocity_raw,
        "present_velocity_rad_s": f"{direction * state.present_velocity_raw * VELOCITY_RAD_S_PER_RAW:.9f}",
        "present_current_raw": state.present_current_raw,
        "present_current_A": f"{current_direction * state.present_current_raw * CURRENT_A_PER_RAW:.9f}",
        "present_pwm_raw": state.present_pwm_raw,
        "present_pwm_fraction": f"{pwm_direction * state.present_pwm_raw * PWM_FRACTION_PER_RAW:.9f}",
        "velocity_trajectory_raw": state.velocity_trajectory_raw,
        "velocity_trajectory_rad_s": f"{direction * state.velocity_trajectory_raw * VELOCITY_RAD_S_PER_RAW:.9f}",
        "position_trajectory_raw": state.position_trajectory_raw,
        "position_trajectory_rad": f"{cfg.raw_to_rad(state.position_trajectory_raw):.9f}",
        "input_voltage_raw": state.input_voltage_raw, "input_voltage_V": f"{state.input_voltage_raw * 0.1:.3f}",
        "temperature_C": state.temperature_c, "moving": state.moving, "moving_status": state.moving_status,
        "current_saturated": int(abs(state.present_current_raw) >= current_cap_raw),
        "pwm_saturated": int(abs(state.present_pwm_raw) >= pwm_cap_raw),
        "cycle_overrun_ns": overrun, "validity_tag": "NORMAL",
    }


def _poll_error(bus: CanonicalMode5Bus, writer: csv.DictWriter | None) -> None:
    poll_start = time.monotonic_ns()
    error = bus.read_hardware_error()
    poll_end = time.monotonic_ns()
    if writer is not None:
        writer.writerow({"host_time_ns": poll_end, "poll_start_ns": poll_start, "hardware_error_raw": error})
    if error:
        raise RuntimeError(f"Hardware Error Status={error}")


def _run_samples(bus: CanonicalMode5Bus, cfg: CanonicalCampaign, samples: Iterable[Sample], writer: csv.DictWriter | None, safety_writer: csv.DictWriter | None,
                 telemetry_callback: Callable[[dict[str, object]], None] | None = None,
                 abort_requested: Callable[[], bool] | None = None) -> AcquisitionStats:
    period_ns = round(1e9 / cfg.command_rate_hz)
    error_period_ns = round(1e9 / float(cfg.timing["hardware_error_poll_rate_hz"]))
    next_error_poll = time.monotonic_ns()
    start = time.monotonic_ns()
    unwrap = TickUnwrapper()
    monitor = SafetyMonitor(cfg)
    command_times: list[int] = []
    state_times: list[int] = []
    overruns: list[int] = []
    for sample in samples:
        if abort_requested is not None and abort_requested():
            raise OperatorAbort("operator_abort")
        deadline = start + sample.sample_index * period_ns
        remaining = deadline - time.monotonic_ns()
        if remaining > 0:
            time.sleep(remaining * 1e-9)
        tx0 = time.monotonic_ns()
        bus.write_goal_rad_no_response(sample.goal_position_rad)
        tx1 = time.monotonic_ns()
        state = bus.read_state()  # every cycle; Hardware Error never replaces this sample
        rx = time.monotonic_ns()
        if rx >= next_error_poll:
            _poll_error(bus, safety_writer)
            next_error_poll = rx + error_period_ns
        # Include the separate Hardware Error poll in deadline accounting even
        # though the state RX timestamp itself remains the state-read timestamp.
        overrun = max(0, time.monotonic_ns() - (deadline + period_ns))
        monitor.check(state, sample.goal_position_rad, overrun)
        row = _row(cfg, sample, state, tx0, tx1, rx, overrun,
                   unwrap.update(state.realtime_tick_raw), sample.sample_index, True)
        if writer is not None:
            writer.writerow(row)
        if telemetry_callback is not None:
            telemetry_callback(row)
        command_times.append(tx1)
        state_times.append(rx)
        overruns.append(overrun)
    return AcquisitionStats(len(state_times), tuple(command_times), tuple(state_times), tuple(overruns))


def _run_delay_samples(bus: CanonicalMode5Bus, cfg: CanonicalCampaign, samples: list[Sample],
                       writer: csv.DictWriter, safety_writer: csv.DictWriter,
                       telemetry_callback: Callable[[dict[str, object]], None] | None = None,
                       abort_requested: Callable[[], bool] | None = None) -> AcquisitionStats:
    """Poll state at the delay rate and transmit only ZOH command events."""
    telemetry_rate = float(cfg.timing.get("delay_telemetry_target_rate_hz") or cfg.command_rate_hz)
    period_ns = round(1e9 / telemetry_rate)
    error_period_ns = round(1e9 / float(cfg.timing["hardware_error_poll_rate_hz"]))
    events = command_events(samples)
    if not events:
        raise ValueError("delay command event가 없습니다.")
    duration_sec = samples[-1].scheduled_time_sec + 1.0 / cfg.command_rate_hz
    telemetry_count = max(1, math.ceil(duration_sec * telemetry_rate))
    start = time.monotonic_ns()
    next_error_poll = start
    event_index = -1
    command_seq = -1
    last_tx0 = last_tx1 = start
    current_goal = events[0].goal_position_rad
    command_times: list[int] = []
    state_times: list[int] = []
    overruns: list[int] = []
    unwrap = TickUnwrapper()
    monitor = SafetyMonitor(cfg)
    for sample_index in range(telemetry_count):
        if abort_requested is not None and abort_requested():
            raise OperatorAbort("operator_abort")
        scheduled_sec = sample_index / telemetry_rate
        deadline = start + sample_index * period_ns
        remaining = deadline - time.monotonic_ns()
        if remaining > 0:
            time.sleep(remaining * 1e-9)
        command_event = False
        while event_index + 1 < len(events) and events[event_index + 1].scheduled_time_sec <= scheduled_sec + 1e-12:
            event_index += 1
            command_seq += 1
            current_goal = events[event_index].goal_position_rad
            last_tx0 = time.monotonic_ns()
            bus.write_goal_rad_no_response(current_goal)
            last_tx1 = time.monotonic_ns()
            command_times.append(last_tx1)
            command_event = True
        state = bus.read_state()
        rx = time.monotonic_ns()
        if rx >= next_error_poll:
            _poll_error(bus, safety_writer)
            next_error_poll = rx + error_period_ns
        overrun = max(0, time.monotonic_ns() - (deadline + period_ns))
        monitor.check(state, current_goal, overrun)
        phase = events[max(0, event_index)].phase
        sample = Sample(sample_index, scheduled_sec, phase, current_goal)
        row = _row(cfg, sample, state, last_tx0, last_tx1, rx, overrun,
                   unwrap.update(state.realtime_tick_raw), command_seq, command_event)
        writer.writerow(row)
        if telemetry_callback is not None:
            telemetry_callback(row)
        state_times.append(rx)
        overruns.append(overrun)
    return AcquisitionStats(len(state_times), tuple(command_times), tuple(state_times), tuple(overruns))


def run_directory(cfg: CanonicalCampaign, relative: str) -> Path:
    if cfg.campaign_id is None:
        raise ValueError("campaign.id가 미확정입니다.")
    return cfg.output_root / cfg.campaign_id / relative


def collect_run(cfg: CanonicalCampaign, experiment: str, relative: str, mechanical_configuration: str,
                trajectory: str, repeat: int, samples: list[Sample],
                order_override_reason: str | None = None,
                physical_setup_confirmation: dict[str, object] | None = None,
                telemetry_callback: Callable[[dict[str, object]], None] | None = None,
                abort_requested: Callable[[], bool] | None = None,
                bus_factory: type[CanonicalMode5Bus] = CanonicalMode5Bus,
                per_run_override: dict[str, object] | None = None) -> Path:
    missing = cfg.common_execution_missing() if experiment == "manual" else cfg.execution_missing(experiment)
    if missing:
        raise RuntimeError("실기체 실행 전 미확정 항목:\n" + "\n".join(f"- {item}" for item in missing))
    logical = ((cfg.project_root / "data/temp/manual" / relative).resolve()
               if experiment == "manual" else run_directory(cfg, relative))
    target, attempt_index, retry_of = allocate_attempt(logical)
    metadata: dict[str, object] = {
        "valid_flag": False, "invalid_reason": "collection_not_completed",
        "campaign_id": cfg.campaign_id, "experiment": experiment,
        "relative_path": relative, "logical_run_id": relative,
        "attempt_index": attempt_index, "retry_of": retry_of,
        "mechanical_configuration": mechanical_configuration, "trajectory": trajectory,
        "repeat": repeat,
        "split_role": (
            "static_calibration" if experiment == "static" else
            "delay_calibration" if experiment == "delay" else
            "validation" if experiment == "collect" and mechanical_configuration == cfg.holdout_configuration else
            "fit" if experiment == "collect" else
            "manual_temporary" if experiment == "manual" else "pilot"
        ),
        "execution_order_override_reason": order_override_reason,
        "physical_setup_confirmation": physical_setup_confirmation,
        "per_run_override": per_run_override,
        "started_at": datetime.now().astimezone().isoformat(), "config_manifest": cfg.config_manifest(),
        "resolved": {"hardware": cfg.hardware, "timing": cfg.timing, "controller": cfg.registers,
                     "safety": cfg.safety, "geometry": cfg.geometry, "loads": cfg.loads,
                     "trajectory": (per_run_override if experiment == "manual" else cfg.trajectories.get(trajectory, cfg.pilot))},
        "software": {"python": platform.python_version(), "platform": platform.platform(),
                     "git_commit": _git_commit(cfg.project_root)},
    }
    metadata_path = target / "metadata.json"
    try:
        with bus_factory(cfg) as bus:
            model_number = bus.ping()
            if model_number != int(cfg.hardware["expected_model_number"]):
                raise RuntimeError(f"model number 불일치: expected={cfg.hardware['expected_model_number']}, actual={model_number}")
            bus.configure_and_verify()
            firmware = bus.read("firmware_version")
            initial = bus.read_state()
            q_initial = cfg.raw_to_rad(initial.present_position_raw)
            target_initial = samples[0].goal_position_rad
            bus.write_goal_rad(q_initial)
            bus.torque(True)
            try:
                bus.arm_bus_watchdog()
                readback = bus.read_configuration_snapshot()
                _run_samples(bus, cfg, _transition(q_initial, target_initial, cfg), None, None)
                start_state = bus.read_state()
                with (target / "telemetry.csv").open("x", newline="") as stream, (target / "safety_events.csv").open("x", newline="") as safety_stream:
                    writer = csv.DictWriter(stream, fieldnames=TELEMETRY_FIELDS)
                    writer.writeheader()
                    safety_writer = csv.DictWriter(safety_stream, fieldnames=("host_time_ns", "poll_start_ns", "hardware_error_raw"))
                    safety_writer.writeheader()
                    if experiment == "delay":
                        stats = _run_delay_samples(bus, cfg, samples, writer, safety_writer,
                                                   telemetry_callback, abort_requested)
                    else:
                        stats = _run_samples(bus, cfg, samples, writer, safety_writer,
                                             telemetry_callback, abort_requested)
                end_state = bus.read_state()
            finally:
                bus.torque(False)
            metadata.update({
                "model_number": model_number, "firmware_version": firmware, "register_readback": readback,
                **timing_statistics(stats), "temperature_start_C": start_state.temperature_c,
                "temperature_end_C": end_state.temperature_c,
                "timestamp_contract": {
                    "command_tx_before_ns": "host monotonic time immediately before GroupSyncWrite.txPacket",
                    "command_tx_after_ns": "host monotonic time immediately after syncWriteTxOnly returns; no Status Packet awaited",
                    "host_time_ns": "host monotonic time immediately after state GroupSyncRead returns",
                },
                "warmup_procedure": cfg.safety["warmup_procedure"],
                "warmup_acknowledged_at": cfg.approval.get("warmup_acknowledged_at"),
            })
    except BaseException as exc:
        metadata["invalid_reason"] = repr(exc)
        metadata["operator_abort"] = isinstance(exc, OperatorAbort)
        metadata["finished_at"] = datetime.now().astimezone().isoformat()
        _write_json_exclusive(metadata_path, metadata)
        raise
    metadata["valid_flag"] = True
    metadata["invalid_reason"] = ""
    metadata["finished_at"] = datetime.now().astimezone().isoformat()
    _write_json_exclusive(metadata_path, metadata)
    return target
