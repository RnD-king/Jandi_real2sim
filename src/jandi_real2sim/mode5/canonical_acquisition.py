"""Immutable raw-data acquisition for the README-v3 Mode-5 bench."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from .canonical_bus import CanonicalMode5Bus, GoalWrite, State, TimedState
from .canonical_attempts import allocate_attempt
from .canonical_config import CanonicalCampaign
from .canonical_trajectories import Sample
from .canonical_trajectories import command_events
from .spec import REALTIME_TICK_MODULUS


CURRENT_A_PER_RAW = 0.00336
PWM_FRACTION_PER_RAW = 0.00113
VELOCITY_RAD_S_PER_RAW = 0.229 * 2.0 * math.pi / 60.0

TELEMETRY_FIELDS = (
    "sample_index", "scheduled_time_sec", "phase",
    "target_seq", "target_update_event", "bus_write_seq", "bus_write_event",
    "command_seq", "command_event", "command_tx_before_ns", "command_tx_after_ns",
    "state_read_before_ns", "state_read_after_ns", "state_time_ns",
    "host_time_ns", "host_time_sec",
    "goal_position_raw", "goal_position_rad", "goal_position_readback_raw",
    "goal_position_readback_rad", "realtime_tick_raw",
    "realtime_tick_unwrapped_ms", "present_position_raw", "present_position_rad",
    "present_velocity_raw", "present_velocity_rad_s", "present_current_raw",
    "present_current_A", "present_pwm_raw", "present_pwm_fraction",
    "velocity_trajectory_raw", "velocity_trajectory_rad_s",
    "position_trajectory_raw", "position_trajectory_rad", "input_voltage_raw",
    "input_voltage_V", "temperature_C", "moving", "moving_status",
    "current_saturated", "pwm_saturated", "current_near_limit", "pwm_near_limit",
    "goal_readback_mismatch", "timing_invalid", "cycle_overrun_ns",
    "fit_eligible", "validity_tag",
)

SAFETY_FIELDS = (
    "host_time_ns", "event", "poll_start_ns", "poll_end_ns", "poll_duration_ns",
    "hardware_error_raw", "detail",
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


def _git_dirty(root: Path) -> bool | None:
    try:
        return bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _manifest_sha256(manifest: dict[str, dict[str, str]]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _critical_config_manifest(cfg: CanonicalCampaign) -> dict[str, dict[str, str]]:
    """YAML experiment inputs only; documentation edits do not alter physics."""
    return {role: value for role, value in cfg.config_manifest().items() if role != "README"}


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
    target_update_ns: tuple[int, ...]
    state_time_ns: tuple[int, ...]
    overruns_ns: tuple[int, ...]
    bus_write_ns: tuple[int, ...] = ()
    state_read_durations_ns: tuple[int, ...] = ()
    goal_readback_mismatch_count: int = 0
    severe_overrun_count: int = 0

    @property
    def command_tx_after_ns(self) -> tuple[int, ...]:
        return self.target_update_ns

    @property
    def state_rx_ns(self) -> tuple[int, ...]:
        return self.state_time_ns


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
    result.update(interval(stats.target_update_ns, "target_update"))
    result.update(interval(stats.bus_write_ns or stats.target_update_ns, "bus_write"))
    result.update(interval(stats.state_time_ns, "state"))
    result["measured_command_rate_hz"] = result["measured_target_update_rate_hz"]
    positive = [value for value in stats.overruns_ns if value > 0]
    result["deadline_overrun_count"] = len(positive)
    result["deadline_overrun_max_ns"] = max(positive, default=0)
    result["state_read_duration_median_ms"] = (
        float(sorted(stats.state_read_durations_ns)[len(stats.state_read_durations_ns) // 2] * 1e-6)
        if stats.state_read_durations_ns else 0.0
    )
    result["goal_readback_mismatch_count"] = stats.goal_readback_mismatch_count
    result["severe_overrun_count"] = stats.severe_overrun_count
    if len(stats.state_time_ns) >= 2:
        intervals = [right - left for left, right in zip(stats.state_time_ns, stats.state_time_ns[1:])]
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


def _write_timed(bus: CanonicalMode5Bus, goal: float) -> GoalWrite:
    before = time.monotonic_ns()
    result = bus.write_goal_rad_no_response(goal)
    after = time.monotonic_ns()
    return result if isinstance(result, GoalWrite) else GoalWrite(0, before, after)


def _read_timed(bus: CanonicalMode5Bus) -> TimedState:
    method = getattr(bus, "read_state_timed", None)
    if callable(method):
        result = method()
        if isinstance(result, TimedState):
            return result
    before = time.monotonic_ns()
    state = bus.read_state()
    after = time.monotonic_ns()
    return TimedState(state, before, after)


def _row(cfg: CanonicalCampaign, sample_index: int, sample: Sample, state: State,
         write: GoalWrite, state_timing: TimedState, overrun: int, tick_ms: int,
         target_seq: int, target_update: bool, bus_write_seq: int,
         readback_mismatch: bool, timing_invalid: bool, bus_write_event: bool = True) -> dict[str, object]:
    direction = int(cfg.hardware["direction"])
    current_direction = int(cfg.hardware["current_direction"])
    pwm_direction = int(cfg.hardware["pwm_direction"])
    current_cap_raw = min(abs(int(cfg.registers["goal_current_raw"])), int(cfg.registers["expected_current_limit_raw"]))
    pwm_cap_raw = min(abs(int(cfg.registers["goal_pwm_raw"])), int(cfg.registers["expected_pwm_limit_raw"]))
    current_saturated = abs(state.present_current_raw) >= current_cap_raw
    pwm_saturated = abs(state.present_pwm_raw) >= pwm_cap_raw
    current_near = cfg.timing.get("current_near_limit_fraction")
    pwm_near = cfg.timing.get("pwm_near_limit_fraction")
    state_time = (state_timing.read_before_ns + state_timing.read_after_ns) // 2
    tags = []
    if timing_invalid: tags.append("TIMING_INVALID")
    if readback_mismatch: tags.append("GOAL_READBACK_MISMATCH")
    if current_saturated: tags.append("CURRENT_SATURATED")
    if pwm_saturated: tags.append("PWM_SATURATED")
    return {
        "sample_index": sample_index,
        "target_seq": target_seq, "target_update_event": int(target_update),
        "bus_write_seq": bus_write_seq, "bus_write_event": int(bus_write_event),
        # Backward compatibility: command means a new 50 Hz target, not a repeat write.
        "command_seq": target_seq, "command_event": int(target_update),
        "command_tx_before_ns": write.tx_before_ns, "command_tx_after_ns": write.tx_after_ns,
        "state_read_before_ns": state_timing.read_before_ns,
        "state_read_after_ns": state_timing.read_after_ns, "state_time_ns": state_time,
        "host_time_ns": state_timing.read_after_ns,
        "host_time_sec": f"{state_timing.read_after_ns * 1e-9:.9f}",
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
        "current_saturated": int(current_saturated), "pwm_saturated": int(pwm_saturated),
        "current_near_limit": int(current_near is not None and abs(state.present_current_raw) >= float(current_near) * current_cap_raw),
        "pwm_near_limit": int(pwm_near is not None and abs(state.present_pwm_raw) >= float(pwm_near) * pwm_cap_raw),
        "goal_readback_mismatch": int(readback_mismatch), "timing_invalid": int(timing_invalid),
        "cycle_overrun_ns": overrun,
        "fit_eligible": int(not timing_invalid and not readback_mismatch),
        "validity_tag": "|".join(tags) if tags else "NORMAL",
    }


def _poll_error(bus: CanonicalMode5Bus, writer: csv.DictWriter | None) -> None:
    poll_start = time.monotonic_ns()
    error = bus.read_hardware_error()
    poll_end = time.monotonic_ns()
    if writer is not None:
        writer.writerow({"host_time_ns": poll_end, "event": "hardware_error_poll",
                         "poll_start_ns": poll_start, "poll_end_ns": poll_end,
                         "poll_duration_ns": poll_end - poll_start,
                         "hardware_error_raw": error, "detail": ""})
    if error:
        raise RuntimeError(f"Hardware Error Status={error}")


def _run_samples(bus: CanonicalMode5Bus, cfg: CanonicalCampaign, samples: Iterable[Sample], writer: csv.DictWriter | None, safety_writer: csv.DictWriter | None,
                 telemetry_callback: Callable[[dict[str, object]], None] | None = None,
                 abort_requested: Callable[[], bool] | None = None) -> AcquisitionStats:
    planned = list(samples)
    bus_rate = cfg.bus_write_rate_hz
    target_rate = cfg.target_generation_rate_hz
    period_ns = round(1e9 / bus_rate)
    ratio = bus_rate / target_rate
    if not ratio.is_integer():
        raise ValueError("bus_write_rate_hz는 target_generation_rate_hz의 정수배여야 합니다.")
    repeats = int(ratio)
    error_period_ns = round(1e9 / float(cfg.timing["hardware_error_poll_rate_hz"]))
    next_error_poll = time.monotonic_ns() + error_period_ns
    start = time.monotonic_ns()
    unwrap = TickUnwrapper()
    monitor = SafetyMonitor(cfg)
    target_times: list[int] = []
    write_times: list[int] = []
    state_times: list[int] = []
    state_durations: list[int] = []
    overruns: list[int] = []
    mismatch_count = 0
    severe_count = 0
    severe_ns = round(float(cfg.timing["severe_overrun_threshold_sec"]) * 1e9)
    for cycle_index in range(len(planned) * repeats):
        sample = planned[cycle_index // repeats]
        target_update = cycle_index % repeats == 0
        if abort_requested is not None and abort_requested():
            raise OperatorAbort("operator_abort")
        deadline = start + cycle_index * period_ns
        remaining = deadline - time.monotonic_ns()
        if remaining > 0:
            time.sleep(remaining * 1e-9)
        write = _write_timed(bus, sample.goal_position_rad)
        timed_state = _read_timed(bus)  # every cycle; Hardware Error never replaces this sample
        state = timed_state.state
        if timed_state.read_after_ns >= next_error_poll:
            _poll_error(bus, safety_writer)
            next_error_poll += error_period_ns
        # Include the separate Hardware Error poll in deadline accounting even
        # though the state RX timestamp itself remains the state-read timestamp.
        overrun = max(0, time.monotonic_ns() - (deadline + period_ns))
        timing_invalid = overrun >= severe_ns
        expected_raw = cfg.rad_to_raw(sample.goal_position_rad)
        # One repeated-write cycle is the explicit readback grace period.
        readback_mismatch = (not target_update and state.goal_position_raw != expected_raw)
        mismatch_count += int(readback_mismatch)
        monitor.check(state, sample.goal_position_rad, overrun)
        row = _row(cfg, cycle_index, sample, state, write, timed_state, overrun,
                   unwrap.update(state.realtime_tick_raw), sample.sample_index,
                   target_update, cycle_index, readback_mismatch, timing_invalid)
        if writer is not None:
            writer.writerow(row)
        if telemetry_callback is not None:
            telemetry_callback(row)
        if target_update: target_times.append(write.tx_after_ns)
        write_times.append(write.tx_after_ns)
        state_times.append((timed_state.read_before_ns + timed_state.read_after_ns) // 2)
        state_durations.append(timed_state.read_after_ns - timed_state.read_before_ns)
        overruns.append(overrun)
        if readback_mismatch:
            raise RuntimeError("GOAL_READBACK_MISMATCH after one-cycle grace")
        if timing_invalid:
            severe_count += 1
            raise RuntimeError(f"TIMING_INVALID severe overrun={overrun} ns")
    return AcquisitionStats(len(state_times), tuple(target_times), tuple(state_times), tuple(overruns),
                            tuple(write_times), tuple(state_durations), mismatch_count, severe_count)


def _run_delay_samples(bus: CanonicalMode5Bus, cfg: CanonicalCampaign, samples: list[Sample],
                       writer: csv.DictWriter, safety_writer: csv.DictWriter,
                       telemetry_callback: Callable[[dict[str, object]], None] | None = None,
                       abort_requested: Callable[[], bool] | None = None) -> AcquisitionStats:
    """Poll state at the delay rate and transmit only ZOH command events."""
    telemetry_rate = float(cfg.timing.get("delay_telemetry_target_rate_hz") or cfg.state_read_rate_hz)
    period_ns = round(1e9 / telemetry_rate)
    error_period_ns = round(1e9 / float(cfg.timing["hardware_error_poll_rate_hz"]))
    events = command_events(samples)
    if not events:
        raise ValueError("delay command event가 없습니다.")
    duration_sec = samples[-1].scheduled_time_sec + 1.0 / cfg.target_generation_rate_hz
    telemetry_count = max(1, math.ceil(duration_sec * telemetry_rate))
    start = time.monotonic_ns()
    next_error_poll = start + error_period_ns
    event_index = -1
    command_seq = -1
    last_write = GoalWrite(0, start, start)
    current_goal = events[0].goal_position_rad
    command_times: list[int] = []
    state_times: list[int] = []
    state_durations: list[int] = []
    overruns: list[int] = []
    unwrap = TickUnwrapper()
    monitor = SafetyMonitor(cfg)
    severe_ns = round(float(cfg.timing["severe_overrun_threshold_sec"]) * 1e9)
    search_window_ns = round(float(cfg.trajectories["delay_probe"]["response_search_sec"]) * 1e9)
    defer_error_until = start
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
            last_write = _write_timed(bus, current_goal)
            command_times.append(last_write.tx_after_ns)
            defer_error_until = last_write.tx_after_ns + search_window_ns
            command_event = True
        timed_state = _read_timed(bus)
        state = timed_state.state
        state_mid = (timed_state.read_before_ns + timed_state.read_after_ns) // 2
        if timed_state.read_after_ns >= next_error_poll and timed_state.read_after_ns >= defer_error_until:
            _poll_error(bus, safety_writer)
            next_error_poll = timed_state.read_after_ns + error_period_ns
        overrun = max(0, time.monotonic_ns() - (deadline + period_ns))
        timing_invalid = overrun >= severe_ns
        monitor.check(state, current_goal, overrun)
        phase = events[max(0, event_index)].phase
        sample = Sample(sample_index, scheduled_sec, phase, current_goal)
        mismatch = (not command_event and state.goal_position_raw != cfg.rad_to_raw(current_goal))
        row = _row(cfg, sample_index, sample, state, last_write, timed_state, overrun,
                   unwrap.update(state.realtime_tick_raw), command_seq, command_event,
                   command_seq, mismatch, timing_invalid, command_event)
        writer.writerow(row)
        if telemetry_callback is not None:
            telemetry_callback(row)
        state_times.append(state_mid)
        state_durations.append(timed_state.read_after_ns - timed_state.read_before_ns)
        overruns.append(overrun)
        if mismatch:
            raise RuntimeError("GOAL_READBACK_MISMATCH after one-cycle grace")
        if timing_invalid:
            raise RuntimeError(f"TIMING_INVALID severe overrun={overrun} ns")
    return AcquisitionStats(len(state_times), tuple(command_times), tuple(state_times), tuple(overruns),
                            tuple(command_times), tuple(state_durations), 0, 0)


def run_directory(cfg: CanonicalCampaign, relative: str) -> Path:
    if cfg.campaign_id is None:
        raise ValueError("campaign.id가 미확정입니다.")
    return cfg.output_root / cfg.campaign_id / relative


def post_run_integrity_check(path: Path, cfg: CanonicalCampaign, stats: AcquisitionStats,
                             expected_sample_count: int | None = None) -> dict[str, object]:
    """Validate immutable raw after collection; it is the only path to valid_flag=true."""
    errors: list[str] = []
    with (path / "telemetry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != stats.sample_count:
        errors.append(f"sample_count expected={stats.sample_count} actual={len(rows)}")
    if expected_sample_count is not None and len(rows) != expected_sample_count:
        errors.append(f"planned_sample_count expected={expected_sample_count} actual={len(rows)}")
    if [int(row["sample_index"]) for row in rows] != list(range(len(rows))):
        errors.append("sample_index continuity")
    for name in ("host_time_ns", "state_time_ns", "state_read_before_ns", "state_read_after_ns"):
        values = [int(row[name]) for row in rows]
        if any(right <= left for left, right in zip(values, values[1:])):
            errors.append(f"{name} monotonicity")
    numeric = ("goal_position_rad", "present_position_rad", "present_velocity_rad_s",
               "present_current_A", "present_pwm_fraction", "input_voltage_V")
    if any(not math.isfinite(float(row[name])) for row in rows for name in numeric):
        errors.append("non-finite telemetry")
    target_seq = [int(row["target_seq"]) for row in rows if int(row["target_update_event"])]
    if target_seq and target_seq != list(range(target_seq[0], target_seq[0] + len(target_seq))):
        errors.append("target_seq continuity")
    bus_seq = [int(row["bus_write_seq"]) for row in rows if int(row["bus_write_event"])]
    if bus_seq and any(right != left + 1 for left, right in zip(bus_seq, bus_seq[1:])):
        errors.append("bus_write_seq continuity")
    if any(int(row["goal_readback_mismatch"]) for row in rows):
        errors.append("Goal Position readback mismatch")
    if any(int(row["timing_invalid"]) for row in rows):
        errors.append("severe timing overrun")
    ticks = [int(row["realtime_tick_unwrapped_ms"]) for row in rows]
    host = [int(row["state_time_ns"]) for row in rows]
    if len(rows) > 1:
        gross = [abs((b - a) - (d - c) * 1e-6) for a, b, c, d in zip(ticks, ticks[1:], host, host[1:])]
        if gross and max(gross) > 100.0:
            errors.append("gross Realtime Tick/host interval inconsistency")
    timing = timing_statistics(stats)
    if not stats.target_update_ns or not stats.bus_write_ns or not stats.state_time_ns:
        errors.append("missing target/bus/state timing series")
    return {"passed": not errors, "errors": errors, "timing": timing}


def _campaign_freeze_path(cfg: CanonicalCampaign) -> Path:
    assert cfg.campaign_id is not None
    return cfg.output_root / cfg.campaign_id / "campaign_freeze.json"


def _verify_campaign_freeze(cfg: CanonicalCampaign) -> None:
    path = _campaign_freeze_path(cfg)
    if not path.exists():
        return
    frozen = json.loads(path.read_text())
    if frozen.get("critical_config_manifest_sha256") != _manifest_sha256(_critical_config_manifest(cfg)):
        raise RuntimeError("canonical campaign이 freeze 이후 변경되었습니다. NEW CAMPAIGN을 만드십시오.")


def _write_campaign_freeze(cfg: CanonicalCampaign, run: Path) -> None:
    path = _campaign_freeze_path(cfg)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(path, {
        "campaign_id": cfg.campaign_id, "first_valid_run": str(run),
        "frozen_at": datetime.now().astimezone().isoformat(),
        "critical_config_manifest_sha256": _manifest_sha256(_critical_config_manifest(cfg)),
        "critical_config_manifest": _critical_config_manifest(cfg),
    })


def _verify_physical_setup(cfg: CanonicalCampaign, mechanical: str,
                           confirmation: dict[str, object] | None) -> None:
    if not confirmation or confirmation.get("mechanical_configuration") != mechanical:
        raise PermissionError(
            f"physical setup mismatch: requested={mechanical}, confirmed="
            f"{None if not confirmation else confirmation.get('mechanical_configuration')}"
        )
    expected_mass = cfg.load_mass_kg(mechanical)
    expected_length = cfg.arm_length_m(mechanical)
    confirmed_mass = confirmation.get("measured_mass_kg")
    confirmed_length = confirmation.get("arm_length_m")
    if confirmed_mass is None or not math.isclose(float(confirmed_mass), expected_mass, rel_tol=0.0, abs_tol=1e-12):
        raise PermissionError(f"physical measured mass mismatch: expected={expected_mass}, confirmed={confirmed_mass}")
    if confirmed_length is None or not math.isclose(float(confirmed_length), expected_length, rel_tol=0.0, abs_tol=1e-12):
        raise PermissionError(f"physical arm length mismatch: expected={expected_length}, confirmed={confirmed_length}")


def cooldown_remaining_sec(cfg: CanonicalCampaign) -> float:
    if cfg.campaign_id is None:
        return 0.0
    latest: datetime | None = None
    root = cfg.output_root / cfg.campaign_id
    for path in root.glob("**/metadata.json") if root.exists() else ():
        try:
            value = json.loads(path.read_text()).get("finished_at")
            stamp = datetime.fromisoformat(value) if value else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if stamp is not None and (latest is None or stamp > latest):
            latest = stamp
    if latest is None:
        return 0.0
    elapsed = (datetime.now().astimezone() - latest).total_seconds()
    return max(0.0, float(cfg.safety["between_runs_sec"]) - elapsed)


def collect_run(cfg: CanonicalCampaign, experiment: str, relative: str, mechanical_configuration: str,
                trajectory: str, repeat: int, samples: list[Sample],
                order_override_reason: str | None = None,
                physical_setup_confirmation: dict[str, object] | None = None,
                telemetry_callback: Callable[[dict[str, object]], None] | None = None,
                abort_requested: Callable[[], bool] | None = None,
                bus_factory: type[CanonicalMode5Bus] = CanonicalMode5Bus,
                per_run_override: dict[str, object] | None = None) -> Path:
    if experiment == "pilot" and mechanical_configuration != cfg.pilot.get("mechanical_configuration"):
        raise PermissionError(
            f"pilot mechanical mismatch: configured={cfg.pilot.get('mechanical_configuration')}, "
            f"requested={mechanical_configuration}"
        )
    if experiment == "delay" and mechanical_configuration != cfg.trajectories["delay_probe"].get("mechanical_configuration"):
        raise PermissionError(
            f"delay mechanical mismatch: configured={cfg.trajectories['delay_probe'].get('mechanical_configuration')}, "
            f"requested={mechanical_configuration}"
        )
    missing = cfg.common_execution_missing() if experiment == "manual" else cfg.execution_missing(experiment)
    if missing:
        raise RuntimeError("실기체 실행 전 미확정 항목:\n" + "\n".join(f"- {item}" for item in missing))
    if experiment != "manual":
        _verify_physical_setup(cfg, mechanical_configuration, physical_setup_confirmation)
        remaining = cooldown_remaining_sec(cfg)
        if remaining > 0:
            raise RuntimeError(f"Cooldown remaining: {remaining:.1f} s")
    if experiment in ("static", "delay", "collect"):
        _verify_campaign_freeze(cfg)
        if bus_factory is CanonicalMode5Bus and _git_dirty(cfg.project_root):
            raise RuntimeError("git working tree가 dirty여서 canonical hardware collection을 차단했습니다.")
    logical = ((cfg.project_root / "data/temp/manual" / relative).resolve()
               if experiment == "manual" else run_directory(cfg, relative))
    target, attempt_index, retry_of = allocate_attempt(logical)
    manifest = cfg.config_manifest()
    git_dirty = _git_dirty(cfg.project_root)
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
        "started_at": datetime.now().astimezone().isoformat(), "config_manifest": manifest,
        "config_manifest_sha256": _manifest_sha256(manifest),
        "resolved": {"hardware": cfg.hardware, "timing": cfg.timing, "controller": cfg.registers,
                     "safety": cfg.safety, "geometry": cfg.geometry, "loads": cfg.loads,
                     "trajectory": (per_run_override if experiment == "manual" else cfg.trajectories.get(trajectory, cfg.pilot))},
        "software": {"python": platform.python_version(), "platform": platform.platform(),
                     "git_commit": _git_commit(cfg.project_root), "git_dirty": git_dirty},
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
                if experiment == "delay":
                    # Preflight safety read; periodic polls are deferred away from response windows.
                    _poll_error(bus, None)
                _run_samples(bus, cfg, _transition(q_initial, target_initial, cfg), None, None)
                start_state = bus.read_state()
                with (target / "telemetry.csv").open("x", newline="") as stream, (target / "safety_events.csv").open("x", newline="") as safety_stream:
                    writer = csv.DictWriter(stream, fieldnames=TELEMETRY_FIELDS)
                    writer.writeheader()
                    safety_writer = csv.DictWriter(safety_stream, fieldnames=SAFETY_FIELDS)
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
            if experiment == "delay":
                rate = float(cfg.timing.get("delay_telemetry_target_rate_hz") or cfg.state_read_rate_hz)
                duration = samples[-1].scheduled_time_sec + 1.0 / cfg.target_generation_rate_hz
                expected_samples = max(1, math.ceil(duration * rate))
            else:
                expected_samples = len(samples) * int(cfg.bus_write_rate_hz / cfg.target_generation_rate_hz)
            integrity = post_run_integrity_check(target, cfg, stats, expected_samples)
            metadata.update({
                "model_number": model_number, "firmware_version": firmware, "register_readback": readback,
                **timing_statistics(stats), "temperature_start_C": start_state.temperature_c,
                "temperature_end_C": end_state.temperature_c,
                "timestamp_contract": {
                    "command_tx_before_ns": "host monotonic time immediately before GroupSyncWrite.txPacket API call",
                    "command_tx_after_ns": "host monotonic time immediately after syncWriteTxOnly API returns; not RS-485 wire time",
                    "state_read_before_ns": "host monotonic time immediately before GroupSyncRead.txRxPacket API call",
                    "state_read_after_ns": "host monotonic time immediately after GroupSyncRead.txRxPacket API returns",
                    "state_time_ns": "midpoint of host-side state API bracket; not firmware current-update time",
                    "host_time_ns": "compatibility alias for state_read_after_ns",
                },
                "integrity_check": integrity,
                "warmup_procedure": cfg.safety["warmup_procedure"],
                "warmup_acknowledged_at": cfg.approval.get("warmup_acknowledged_at"),
            })
    except BaseException as exc:
        metadata["invalid_reason"] = repr(exc)
        metadata["operator_abort"] = isinstance(exc, OperatorAbort)
        metadata["finished_at"] = datetime.now().astimezone().isoformat()
        safety_path = target / "safety_events.csv"
        if safety_path.exists():
            with safety_path.open("a", newline="") as stream:
                csv.DictWriter(stream, fieldnames=SAFETY_FIELDS).writerow({
                    "host_time_ns": time.monotonic_ns(),
                    "event": "operator_abort" if isinstance(exc, OperatorAbort) else "run_abort",
                    "poll_start_ns": "", "poll_end_ns": "", "poll_duration_ns": "",
                    "hardware_error_raw": "", "detail": repr(exc),
                })
        _write_json_exclusive(metadata_path, metadata)
        raise
    if not bool(metadata["integrity_check"]["passed"]):
        metadata["invalid_reason"] = "; ".join(metadata["integrity_check"]["errors"])
        metadata["finished_at"] = datetime.now().astimezone().isoformat()
        _write_json_exclusive(metadata_path, metadata)
        raise RuntimeError(f"post-run integrity FAIL: {metadata['invalid_reason']}")
    metadata["valid_flag"] = True
    metadata["invalid_reason"] = ""
    metadata["finished_at"] = datetime.now().astimezone().isoformat()
    _write_json_exclusive(metadata_path, metadata)
    if experiment in ("static", "delay", "collect"):
        _write_campaign_freeze(cfg, target)
    return target
