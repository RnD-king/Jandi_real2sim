"""Immutable 100 Hz acquisition for the seven-condition Mode-3 campaign."""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from .bus import Mode3Bus, State
from .config import Campaign
from .trajectories import Sample, build


CURRENT_A_PER_RAW = 0.00336
PWM_FRACTION_PER_RAW = 0.00113
VELOCITY_RAD_S_PER_RAW = 0.229 * 2.0 * math.pi / 60.0
FIELDS = (
    "sample_index", "scheduled_time_sec", "host_time_sec", "phase", "torque_enable",
    "goal_position_raw", "goal_position_rad", "goal_position_readback_raw",
    "present_position_raw", "present_position_rad", "present_velocity_raw",
    "present_velocity_rad_s", "present_pwm_raw", "present_pwm_fraction",
    "present_current_raw", "present_current_A", "input_voltage_V", "temperature_C",
    "command_tx_before_ns", "command_tx_after_ns", "state_read_before_ns",
    "state_read_after_ns", "realtime_tick_raw", "moving", "moving_status",
)
COMMAND_FIELDS = (
    "sample_index", "scheduled_time_sec", "host_time_sec", "phase", "event",
    "torque_enable", "goal_position_raw", "goal_position_rad",
    "command_tx_before_ns", "command_tx_after_ns",
)


def _run_root(cfg: Campaign, condition: str, trajectory: str, repeat: int,
              run_group: str | None = None) -> Path:
    base = cfg.output_root / str(cfg.campaign_id)
    if run_group:
        base = base / run_group
    logical = base / condition / trajectory / f"repeat_{repeat}"
    logical.mkdir(parents=True, exist_ok=True)
    indexes = [int(path.name.split("_")[-1]) for path in logical.glob("attempt_*")
               if path.name.split("_")[-1].isdigit()]
    target = logical / f"attempt_{max(indexes, default=0) + 1:03d}"
    target.mkdir()
    return target


def _values(cfg: Campaign, state: State) -> dict[str, float | int]:
    return {
        "goal_position_readback_raw": state.goal_position_raw,
        "present_position_raw": state.present_position_raw,
        "present_position_rad": cfg.raw_to_rad(state.present_position_raw),
        "present_velocity_raw": state.present_velocity_raw,
        "present_velocity_rad_s": int(cfg.hardware["direction"]) * state.present_velocity_raw * VELOCITY_RAD_S_PER_RAW,
        "present_pwm_raw": state.present_pwm_raw,
        "present_pwm_fraction": int(cfg.hardware["pwm_direction"]) * state.present_pwm_raw * PWM_FRACTION_PER_RAW,
        "present_current_raw": state.present_current_raw,
        "present_current_A": int(cfg.hardware["current_direction"]) * state.present_current_raw * CURRENT_A_PER_RAW,
        "input_voltage_V": state.input_voltage_raw * 0.1,
        "temperature_C": state.temperature_c,
        "realtime_tick_raw": state.realtime_tick_raw,
        "moving": state.moving,
        "moving_status": state.moving_status,
    }


def _check_safety(cfg: Campaign, state: State, goal: float, torque_enabled: bool) -> None:
    row = _values(cfg, state)
    q = float(row["present_position_rad"]); dq = float(row["present_velocity_rad_s"])
    current = float(row["present_current_A"]); pwm = float(row["present_pwm_fraction"])
    voltage = float(row["input_voltage_V"]); temperature = float(row["temperature_C"])
    safety = cfg.safety
    violated = []
    if not float(safety["software_position_min_rad"]) <= q <= float(safety["software_position_max_rad"]): violated.append("position")
    if torque_enabled and abs(goal - q) >= float(safety["maximum_abs_position_error_rad"]): violated.append("position_error")
    if abs(dq) >= float(safety["maximum_abs_velocity_rad_s"]): violated.append("velocity")
    if abs(current) >= float(safety["maximum_abs_current_A"]): violated.append("current")
    if abs(pwm) >= float(safety["maximum_abs_pwm_fraction"]): violated.append("pwm")
    if not float(safety["minimum_input_voltage_v"]) <= voltage <= float(safety["maximum_input_voltage_v"]): violated.append("voltage")
    if temperature >= float(safety["maximum_temperature_c"]): violated.append("temperature")
    if violated:
        raise RuntimeError(f"LIVE SAFETY {','.join(violated)} q={q:.5f} dq={dq:.5f} I={current:.3f} pwm={pwm:.3f}")


def _validate_samples(cfg: Campaign, samples: list[Sample]) -> None:
    if not samples:
        raise ValueError("빈 trajectory는 실행할 수 없습니다.")
    lower = float(cfg.safety["software_position_min_rad"])
    upper = float(cfg.safety["software_position_max_rad"])
    for sample in samples:
        if not math.isfinite(sample.goal_rad):
            raise ValueError(f"비정상 목표각: sample={sample.index}, goal={sample.goal_rad}")
        if not lower <= sample.goal_rad <= upper:
            raise ValueError(
                f"목표각이 software limit 밖입니다: sample={sample.index}, "
                f"goal={sample.goal_rad:.6f}, limits=[{lower:.6f}, {upper:.6f}]"
            )


def _check_hardware_error(bus: Mode3Bus) -> None:
    error = bus.read_hardware_error()
    if error:
        raise RuntimeError(f"DYNAMIXEL Hardware Error Status={error:#04x}")


def _transition(bus: Mode3Bus, cfg: Campaign, start: float, target: float,
                abort: Callable[[], bool]) -> None:
    duration = float(cfg.trajectories["transition_duration_sec"])
    count = max(2, round(duration * cfg.command_rate_hz))
    origin = time.monotonic()
    error_period = 1.0 / float(cfg.timing["hardware_error_poll_rate_hz"])
    next_error_check = origin
    for index in range(count):
        if abort():
            raise InterruptedError("operator abort")
        ratio = index / (count - 1)
        blend = 0.5 - 0.5 * math.cos(math.pi * ratio)
        goal = start + blend * (target - start)
        bus.write_goal_rad_no_response(goal)
        state = bus.read_state()
        _check_safety(cfg, state, goal, True)
        now = time.monotonic()
        if now >= next_error_check:
            _check_hardware_error(bus)
            next_error_check = now + error_period
        time.sleep(max(0.0, origin + (index + 1) / cfg.command_rate_hz - time.monotonic()))


def _plot(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (path / "telemetry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    t = [float(row["host_time_sec"]) for row in rows]
    qg = [float(row["goal_position_rad"]) for row in rows]
    q = [float(row["present_position_rad"]) for row in rows]
    pwm = [float(row["present_pwm_fraction"]) for row in rows]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axes[0].plot(t, qg, "k--", label="q_cmd"); axes[0].plot(t, q, label="real")
    axes[0].set_ylabel("angle [rad]"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(t, pwm); axes[1].set_ylabel("PWM duty"); axes[1].set_xlabel("time [s]"); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path / "trajectory.png", dpi=150); plt.close(fig)


def collect(cfg: Campaign, condition_id: str, trajectory: str, repeat: int,
            *, telemetry: Callable[[dict[str, object]], None] | None = None,
            abort: Callable[[], bool] = lambda: False,
            run_group: str | None = None, role_override: str | None = None) -> Path:
    if cfg.campaign_id is None:
        raise ValueError("campaign.id를 먼저 지정하십시오.")
    if repeat not in cfg.repetitions:
        raise ValueError(f"repeat는 {cfg.repetitions} 중 하나여야 합니다.")
    condition = cfg.condition(condition_id)
    samples = build(cfg, trajectory)
    _validate_samples(cfg, samples)
    path = _run_root(cfg, condition_id, trajectory, repeat, run_group)
    metadata = {
        "schema_version": 1, "campaign_id": cfg.campaign_id, "condition": condition_id,
        "trajectory": trajectory, "repeat": repeat,
        "role": role_override or ("fit" if repeat in cfg.fit_repetitions else "validation"),
        "run_group": run_group,
        "load_disk_count": condition.disk_count,
        "load_mass_kg": condition.mass_kg,
        "load_intrinsic_inertia_kg_m2": condition.intrinsic_inertia_kg_m2,
        "axis_to_load_com_m": condition.distance_m,
        "load_disk_mass_kg": cfg.bench["load"]["disk_mass_kg"],
        "load_disk_diameter_m": cfg.bench["load"]["disk_diameter_m"],
        "load_fastener_mass_kg": cfg.bench["load"]["fastener_mass_kg"],
        "load_fastener_intrinsic_inertia_assumption":
            cfg.bench["load"]["fastener_intrinsic_inertia_assumption"],
        "arm_attached": condition.loaded,
        "arm_mass_kg": cfg.bench["arm"]["mass_kg"],
        "arm_com_radius_m": cfg.bench["arm"]["com_radius_m"],
        "arm_inertia_about_com_kg_m2": cfg.bench["arm"]["inertia_about_com_kg_m2"],
        "arm_inertia_kg_m2": cfg.bench["arm"]["inertia_about_pivot_kg_m2"],
        "horn_mass_kg": (cfg.bench["loaded_mounting_hardware"]["horn_mass_kg"]
                         if condition.loaded else cfg.bench["no_load_hardware"]["horn_mass_kg"]),
        "horn_fastener_mass_kg": (cfg.bench["loaded_mounting_hardware"]["horn_fastener_mass_kg"]
                                  if condition.loaded else cfg.bench["no_load_hardware"]["horn_fastener_mass_kg"]),
        "mounting_hardware_treatment": (
            cfg.bench["loaded_mounting_hardware"]["treatment"]
            if condition.loaded else "no_load_probe_horn_only"
        ),
        "position_p_gain": 850, "position_i_gain": cfg.registers["position_i_gain"],
        "position_d_gain": cfg.registers["position_d_gain"],
        "started_at": datetime.now().astimezone().isoformat(), "valid": False,
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    error: BaseException | None = None
    try:
        with Mode3Bus(cfg) as bus:
            try:
                with (path / "telemetry.csv").open("x", newline="") as stream, \
                     (path / "command_events.csv").open("x", newline="") as event_stream:
                    model = bus.ping()
                    if model != int(cfg.hardware["expected_model_number"]):
                        raise RuntimeError(f"model mismatch expected={cfg.hardware['expected_model_number']} actual={model}")
                    metadata["registers"] = bus.configure_and_verify()
                    start_state = bus.read_state()
                    start_q = cfg.raw_to_rad(start_state.present_position_raw)
                    _check_safety(cfg, start_state, start_q, False)
                    # Hold the measured position before enabling torque.  This
                    # prevents an immediate jump to the first experiment goal.
                    bus.write_goal_rad(start_q); bus.torque(True); bus.arm_bus_watchdog()
                    _transition(bus, cfg, start_q, samples[0].goal_rad, abort)
                    writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader()
                    event_writer = csv.DictWriter(event_stream, fieldnames=COMMAND_FIELDS); event_writer.writeheader()
                    origin = time.monotonic(); torque_enabled = True
                    error_period = 1.0 / float(cfg.timing["hardware_error_poll_rate_hz"])
                    next_error_check = origin
                    drop_spec = cfg.trajectories["lift_and_drop"] if trajectory == "lift_and_drop" else None
                    drop_caught = False
                    drop_release_start: float | None = None
                    drop_catch_goal: float | None = None
                    recovery_start: float | None = None
                    recovery_origin: float | None = None
                    last_q = start_q
                    for sample in samples:
                        if abort(): raise InterruptedError("operator abort")
                        phase = sample.phase
                        requested_torque = sample.torque_enable
                        runtime_goal = sample.goal_rad
                        if drop_spec is not None and phase == "released":
                            if drop_release_start is None:
                                drop_release_start = sample.time_sec
                            if drop_caught:
                                requested_torque = True
                                runtime_goal = float(drop_catch_goal)
                                phase = "release_catch_hold"
                        elif drop_spec is not None and phase in ("recovery", "recovery_hold"):
                            if recovery_start is None:
                                recovery_start = sample.time_sec
                                recovery_origin = last_q
                                if not drop_caught:
                                    metadata["drop_catch"] = {
                                        "reason": "maximum_release_duration",
                                        "elapsed_sec": float(drop_spec["release_duration_sec"]),
                                        "position_rad": last_q,
                                    }
                            requested_torque = True
                            if phase == "recovery":
                                elapsed = sample.time_sec - recovery_start
                                runtime_goal = (float(recovery_origin) +
                                    (float(drop_spec["center_rad"]) - float(recovery_origin)) *
                                    (0.5 - 0.5 * math.cos(math.pi * min(
                                        1.0, elapsed / float(drop_spec["recovery_duration_sec"])))))
                            else:
                                runtime_goal = float(drop_spec["center_rad"])
                        if requested_torque != torque_enabled:
                            if requested_torque:
                                bus.write_goal_rad(runtime_goal)
                            bus.torque(requested_torque); torque_enabled = requested_torque
                        tx_before = tx_after = 0
                        if torque_enabled:
                            write = bus.write_goal_rad_no_response(runtime_goal)
                            tx_before, tx_after = write.tx_before_ns, write.tx_after_ns
                            goal_raw = write.raw
                        else:
                            goal_raw = cfg.rad_to_raw(runtime_goal)
                        event_writer.writerow({
                            "sample_index": sample.index, "scheduled_time_sec": sample.time_sec,
                            "host_time_sec": time.monotonic() - origin, "phase": phase,
                            "event": "goal_write" if torque_enabled else "torque_off_sample",
                            "torque_enable": int(torque_enabled), "goal_position_raw": goal_raw,
                            "goal_position_rad": runtime_goal,
                            "command_tx_before_ns": tx_before, "command_tx_after_ns": tx_after,
                        })
                        sampled_torque_enabled = torque_enabled
                        timed = bus.read_state_timed()
                        values = _values(cfg, timed.state)
                        q = float(values["present_position_rad"])
                        dq = float(values["present_velocity_rad_s"])
                        _check_safety(cfg, timed.state, runtime_goal, torque_enabled)
                        if (drop_spec is not None and sample.phase == "released" and
                                not drop_caught):
                            trigger = None
                            if q <= float(drop_spec["catch_angle_rad"]):
                                trigger = "angle"
                            elif abs(dq) >= float(drop_spec["catch_abs_velocity_rad_s"]):
                                trigger = "velocity"
                            if trigger is not None:
                                # First command the measured angle, then enable torque.
                                # Subsequent recovery samples move smoothly from here.
                                bus.write_goal_rad(q)
                                bus.torque(True)
                                torque_enabled = True
                                drop_caught = True
                                drop_catch_goal = q
                                metadata["drop_catch"] = {
                                    "reason": trigger,
                                    "elapsed_sec": sample.time_sec - float(drop_release_start),
                                    "position_rad": q,
                                    "velocity_rad_s": dq,
                                }
                        now = time.monotonic()
                        if now >= next_error_check:
                            _check_hardware_error(bus)
                            next_error_check = now + error_period
                        row: dict[str, object] = {
                            "sample_index": sample.index, "scheduled_time_sec": sample.time_sec,
                            "host_time_sec": time.monotonic() - origin, "phase": phase,
                            "torque_enable": int(sampled_torque_enabled), "goal_position_raw": goal_raw,
                            "goal_position_rad": runtime_goal, "command_tx_before_ns": tx_before,
                            "command_tx_after_ns": tx_after, "state_read_before_ns": timed.read_before_ns,
                            "state_read_after_ns": timed.read_after_ns, **values,
                        }
                        writer.writerow(row)
                        if telemetry: telemetry(row)
                        last_q = q
                        time.sleep(max(0.0, origin + (sample.index + 1) / cfg.command_rate_hz - time.monotonic()))
                    metadata["valid"] = True
            finally:
                try:
                    bus.torque(False)
                except BaseException:
                    pass
    except BaseException as exc:
        error = exc; metadata["error"] = repr(exc)
    finally:
        metadata["finished_at"] = datetime.now().astimezone().isoformat()
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        if (path / "telemetry.csv").exists():
            _plot(path)
    if error is not None:
        raise error
    return path
