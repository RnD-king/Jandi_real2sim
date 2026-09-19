"""Run the complete, no-refit Mode-3 M3 follow-up validation.

This module deliberately reads the canonical M3/controller/backlash results and
repeat-3 telemetry without modifying any of them.  It performs:

* 1 ms versus 10 ms equivalent-controller update sensitivity;
* position-derived velocity validation with 5/7/9 point Savitzky-Golay filters;
* canonical M3 baseline aggregation;
* 0/0.5/1.0/1.5x stateful backlash-width sensitivity for two encoder views;
* direction-reversal-local metrics and compact reports.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml
from scipy.signal import savgol_filter

from jandi_real2sim.mode3_bam.config import DEFAULT_CAMPAIGN, Campaign, load_campaign
from jandi_real2sim.mode3_bam.mujoco_validation import (
    LOADED_CONDITIONS,
    LOADED_TRAJECTORIES,
    ReplayRun,
    RigProperties,
    bench_xml,
    load_identified_model,
    load_replay_run,
    rig_properties,
)


PRIMARY_DERIVED_WINDOW = 7
REVERSAL_HALF_WINDOW_SEC = 0.100
REVERSAL_MIN_SEPARATION_SEC = 0.150
REVERSAL_SPEED_THRESHOLD_RAD_S = 0.05
MAXIMUM_BACKLASH_LIMIT_VIOLATION_FRACTION = 0.05


@dataclass(frozen=True)
class FollowupTrace:
    """One MuJoCo trace with explicit encoder and physical-output views."""

    variant: str
    feedback_view: str
    controller_dt_sec: float
    t: np.ndarray
    encoder_q: np.ndarray
    encoder_dq: np.ndarray
    output_q: np.ndarray
    output_dq: np.ndarray
    duty: np.ndarray
    current: np.ndarray
    motor_torque: np.ndarray
    friction_torque: np.ndarray
    backlash_q: np.ndarray
    backlash_width_rad: float
    maximum_backlash_limit_violation_rad: float


def _zoh(t: np.ndarray, values: np.ndarray, query: float) -> float:
    index = int(np.searchsorted(t, query, side="right") - 1)
    return float(values[min(max(index, 0), len(values) - 1)])


def _sample(trace: FollowupTrace, values: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.interp(t, trace.t, values)


def _error_metrics(predicted: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted) - np.asarray(measured)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
    }


def _m3_friction_limit(
    parameters: dict[str, float],
    velocity: float,
    motor_torque: float,
    external_torque: float,
) -> float:
    return max(
        0.0,
        parameters["coulomb_friction_nm"]
        + parameters["viscous_friction_nm_s_per_rad"] * abs(velocity)
        + parameters["load_friction"] * abs(motor_torque - external_torque),
    )


def backlash_bench_xml(
    cfg: Campaign,
    run: ReplayRun,
    armature_kg_m2: float,
    physics_timestep_sec: float,
    total_backlash_width_rad: float,
) -> str:
    """Build an actuated hinge followed by a limited passive backlash hinge."""
    if total_backlash_width_rad <= 0.0:
        raise ValueError("total backlash width must be positive")
    rig: RigProperties = rig_properties(run.metadata)
    half = 0.5 * total_backlash_width_rad
    gravity_sign = float(cfg.bench["gravity_torque_sign"])
    gravity_zero = float(cfg.bench["gravity_zero_angle_rad"])
    if gravity_sign not in (-1.0, 1.0):
        raise ValueError("gravity_torque_sign must be -1 or +1")
    if not math.isclose(gravity_zero, 0.0, abs_tol=1e-12):
        raise ValueError("follow-up bench requires gravity_zero_angle_rad=0")
    if physics_timestep_sec <= 0.0:
        raise ValueError("physics timestep must be positive")
    inertia = rig.inertia_about_com_kg_m2
    visual_length = max(rig.load_radius_m, 2.0 * rig.center_of_mass_m, 0.08)
    disk_radius = 0.5 * float(run.metadata.get("load_disk_diameter_m", 0.07))
    axis_y = gravity_sign
    # Positive-format solref uses timeconstant/dampratio.  Keep the constraint
    # at MuJoCo's recommended 2*dt lower bound; the backlash validation uses a
    # finer physics timestep than the ordinary cadence experiment.
    limit_time_constant = 2.0 * physics_timestep_sec
    return f"""
<mujoco model="jandi_mode3_m3_stateful_backlash">
  <compiler angle="radian" inertiafromgeom="false"/>
  <option timestep="{physics_timestep_sec:.12g}" gravity="0 0 -{float(cfg.bench['gravity_m_s2']):.12g}" integrator="implicitfast"/>
  <visual><global azimuth="135" elevation="-15"/></visual>
  <worldbody>
    <light pos="0 -1 1.5" dir="0 1 -1"/>
    <body name="actuator_carrier" pos="0 0 0">
      <joint name="servo" type="hinge" axis="0 {axis_y:.0f} 0"
             armature="{armature_kg_m2:.12g}" damping="0" frictionloss="0"/>
      <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-9 1e-9 1e-9"/>
      <body name="pendulum" pos="0 0 0">
        <joint name="passive_backlash" type="hinge" axis="0 {axis_y:.0f} 0"
               limited="true" range="{-half:.12g} {half:.12g}"
               armature="1e-8" damping="0" frictionloss="0"
               solreflimit="{limit_time_constant:.12g} 1"
               solimplimit="0.95 0.99 0.001"/>
        <inertial pos="0 0 {rig.center_of_mass_m:.12g}" mass="{rig.physical_mass_kg:.12g}"
                  diaginertia="{inertia:.12g} {inertia:.12g} {inertia:.12g}"/>
        <geom name="arm_visual" type="capsule" fromto="0 0 0 0 0 {visual_length:.12g}"
              size="0.006" rgba="0.35 0.45 0.55 1" contype="0" conaffinity="0"/>
        <geom name="load_visual" type="cylinder" pos="0 0 {rig.load_radius_m:.12g}"
              size="{disk_radius:.12g} 0.012" rgba="0.85 0.55 0.12 1" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()


def simulate_followup(
    cfg: Campaign,
    run: ReplayRun,
    parameters: dict[str, float],
    controller: dict[str, float],
    *,
    controller_dt_sec: float,
    physics_timestep_sec: float = 0.001,
    backlash_width_rad: float = 0.0,
    feedback_view: str = "single_joint",
) -> FollowupTrace:
    """Simulate M3 without refitting, with controller output held between updates."""
    if controller_dt_sec < physics_timestep_sec - 1e-12:
        raise ValueError("controller timestep cannot be smaller than physics timestep")
    backlash = backlash_width_rad > 0.0
    valid_views = {"single_joint"} if not backlash else {"actuator_side", "output_side"}
    if feedback_view not in valid_views:
        raise ValueError(f"feedback_view must be one of {sorted(valid_views)}")
    xml = (
        backlash_bench_xml(
            cfg, run, parameters["armature_kg_m2"], physics_timestep_sec,
            backlash_width_rad,
        )
        if backlash else
        bench_xml(cfg, run, parameters["armature_kg_m2"], physics_timestep_sec)
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[0] = run.q[0]
    data.qvel[0] = run.dq[0]
    if backlash:
        data.qpos[1] = 0.0
        data.qvel[1] = 0.0
    mujoco.mj_forward(model, data)

    duration = float(run.t[-1])
    count = int(math.ceil(duration / physics_timestep_sec)) + 1
    times = np.minimum(np.arange(count, dtype=float) * physics_timestep_sec, duration)
    arrays = {name: np.empty(count) for name in (
        "encoder_q", "encoder_dq", "output_q", "output_dq", "duty", "current",
        "motor_torque", "friction_torque", "backlash_q",
    )}
    delay = float(controller["command_delay_sec"])
    pwm_limit = int(cfg.registers["expected_pwm_limit_raw"]) * 0.00113
    p_gain = float(controller.get("position_p_gain", 850.0))
    gravity_scale = (
        float(run.metadata["load_mass_kg"])
        * float(cfg.bench["gravity_m_s2"])
        * float(run.metadata["axis_to_load_com_m"])
        + float(run.metadata["arm_mass_kg"])
        * float(cfg.bench["gravity_m_s2"])
        * float(run.metadata["arm_com_radius_m"])
    )
    gravity_zero = float(cfg.bench["gravity_zero_angle_rad"])
    gravity_sign = float(cfg.bench["gravity_torque_sign"])
    next_controller_update = 0.0
    held_duty = 0.0
    previous_enabled = False
    max_limit_violation = 0.0
    half_backlash = 0.5 * backlash_width_rad

    for index, now in enumerate(times):
        servo_q = float(data.qpos[0])
        servo_dq = float(data.qvel[0])
        play_q = float(data.qpos[1]) if backlash else 0.0
        play_dq = float(data.qvel[1]) if backlash else 0.0
        output_q = servo_q + play_q
        output_dq = servo_dq + play_dq
        if feedback_view == "actuator_side":
            encoder_q, encoder_dq = servo_q, servo_dq
        elif feedback_view == "output_side":
            encoder_q, encoder_dq = output_q, output_dq
        else:
            encoder_q, encoder_dq = servo_q, servo_dq

        enabled = _zoh(run.t, run.torque_enable.astype(float), now) > 0.5
        if enabled != previous_enabled:
            next_controller_update = now
        if now + 1e-12 >= next_controller_update:
            goal = _zoh(run.t, run.goal, now - delay)
            held_duty = float(np.clip(
                controller["pwm_error_gain_per_raw_rad"] * p_gain * (goal - encoder_q)
                - controller["pwm_velocity_gain_s_per_rad"] * encoder_dq,
                -pwm_limit,
                pwm_limit,
            )) if enabled else 0.0
            steps = max(1, int(math.floor((now + 1e-12) / controller_dt_sec)) + 1)
            next_controller_update = steps * controller_dt_sec
        if not enabled:
            held_duty = 0.0

        voltage = float(np.interp(now, run.t, run.voltage))
        predicted_current = (
            (held_duty * voltage - parameters["kt_nm_per_a"] * servo_dq)
            / parameters["resistance_ohm"]
        ) if enabled else 0.0
        motor_torque = parameters["kt_nm_per_a"] * predicted_current
        external_torque = (
            gravity_sign * gravity_scale * math.sin(output_q - gravity_zero)
        )
        friction_limit = _m3_friction_limit(
            parameters, servo_dq, motor_torque, external_torque
        )
        model.dof_frictionloss[0] = friction_limit
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[0] = motor_torque
        mujoco.mj_forward(model, data)

        arrays["encoder_q"][index] = encoder_q
        arrays["encoder_dq"][index] = encoder_dq
        arrays["output_q"][index] = output_q
        arrays["output_dq"][index] = output_dq
        arrays["duty"][index] = held_duty
        arrays["current"][index] = predicted_current
        arrays["motor_torque"][index] = motor_torque
        arrays["friction_torque"][index] = float(data.qfrc_constraint[0])
        arrays["backlash_q"][index] = play_q
        if backlash:
            max_limit_violation = max(max_limit_violation, abs(play_q) - half_backlash)
        previous_enabled = enabled
        if index + 1 < count:
            mujoco.mj_step(model, data)

    variant = "m3" if not backlash else f"m3_backlash_{feedback_view}"
    return FollowupTrace(
        variant=variant,
        feedback_view=feedback_view,
        controller_dt_sec=controller_dt_sec,
        t=times,
        encoder_q=arrays["encoder_q"],
        encoder_dq=arrays["encoder_dq"],
        output_q=arrays["output_q"],
        output_dq=arrays["output_dq"],
        duty=arrays["duty"],
        current=arrays["current"],
        motor_torque=arrays["motor_torque"],
        friction_torque=arrays["friction_torque"],
        backlash_q=arrays["backlash_q"],
        backlash_width_rad=backlash_width_rad,
        maximum_backlash_limit_violation_rad=max(0.0, max_limit_violation),
    )


def uniform_derived_velocity(
    t: np.ndarray,
    position: np.ndarray,
    *,
    sample_dt_sec: float,
    window_length: int,
    polynomial_order: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample to a uniform grid and derive velocity with one shared pipeline."""
    if window_length % 2 != 1 or window_length <= polynomial_order:
        raise ValueError("Savitzky-Golay window must be odd and greater than polyorder")
    if sample_dt_sec <= 0.0:
        raise ValueError("derived velocity sample timestep must be positive")
    if len(t) < window_length:
        raise ValueError("run is shorter than the derivative window")
    grid = np.arange(0.0, float(t[-1]) + 0.5 * sample_dt_sec, sample_dt_sec)
    uniform_position = np.interp(grid, t, position)
    velocity = savgol_filter(
        uniform_position,
        window_length=window_length,
        polyorder=polynomial_order,
        deriv=1,
        delta=sample_dt_sec,
        mode="interp",
    )
    valid = np.ones(len(grid), dtype=bool)
    edge = window_length // 2
    valid[:edge] = False
    valid[-edge:] = False
    return grid, uniform_position, np.asarray(velocity), valid


def trace_metrics(
    run: ReplayRun,
    trace: FollowupTrace,
    *,
    derived_sample_dt_sec: float,
    derived_window: int,
) -> dict[str, float]:
    sampled_q = _sample(trace, trace.encoder_q, run.t)
    sampled_dq = _sample(trace, trace.encoder_dq, run.t)
    sampled_pwm = _sample(trace, trace.duty, run.t)
    sampled_current = _sample(trace, trace.current, run.t)
    result: dict[str, float] = {}
    for name, predicted, measured in (
        ("position", sampled_q, run.q),
        ("present_velocity", sampled_dq, run.dq),
        ("pwm", sampled_pwm, run.pwm),
        ("current", sampled_current, run.current),
    ):
        metric = _error_metrics(predicted, measured)
        result[f"{name}_mae"] = metric["mae"]
        result[f"{name}_rmse"] = metric["rmse"]

    grid, _, real_derived, valid = uniform_derived_velocity(
        run.t, run.q,
        sample_dt_sec=derived_sample_dt_sec,
        window_length=derived_window,
    )
    _, _, simulated_derived, simulated_valid = uniform_derived_velocity(
        trace.t, trace.encoder_q,
        sample_dt_sec=derived_sample_dt_sec,
        window_length=derived_window,
    )
    if len(simulated_derived) != len(real_derived):
        simulated_derived = np.interp(
            grid,
            np.arange(len(simulated_derived)) * derived_sample_dt_sec,
            simulated_derived,
        )
    valid &= simulated_valid[:len(valid)]
    derived_metric = _error_metrics(simulated_derived[valid], real_derived[valid])
    result["derived_velocity_mae"] = derived_metric["mae"]
    result["derived_velocity_rmse"] = derived_metric["rmse"]
    real_present_uniform = np.interp(grid, run.t, run.dq)
    measurement_metric = _error_metrics(
        real_present_uniform[valid], real_derived[valid]
    )
    result["real_present_vs_derived_velocity_mae"] = measurement_metric["mae"]
    result["real_present_vs_derived_velocity_rmse"] = measurement_metric["rmse"]
    result["maximum_abs_simulated_pwm_fraction"] = float(
        np.max(np.abs(sampled_pwm))
    )
    result["maximum_abs_simulated_current_A"] = float(
        np.max(np.abs(sampled_current))
    )
    result["maximum_backlash_limit_violation_rad"] = float(
        trace.maximum_backlash_limit_violation_rad
    )
    half_width = 0.5 * trace.backlash_width_rad
    result["maximum_backlash_limit_violation_fraction_of_half_width"] = (
        float(trace.maximum_backlash_limit_violation_rad / half_width)
        if half_width > 0.0 else 0.0
    )
    return result


def commanded_reversal_times(
    run: ReplayRun,
    *,
    sample_dt_sec: float,
    window_length: int = PRIMARY_DERIVED_WINDOW,
) -> list[float]:
    grid, _, velocity, valid = uniform_derived_velocity(
        run.t, run.goal,
        sample_dt_sec=sample_dt_sec,
        window_length=window_length,
    )
    sign = np.zeros(len(velocity), dtype=int)
    sign[velocity > REVERSAL_SPEED_THRESHOLD_RAD_S] = 1
    sign[velocity < -REVERSAL_SPEED_THRESHOLD_RAD_S] = -1
    events: list[float] = []
    last_nonzero = 0
    for index, value in enumerate(sign):
        if not valid[index] or value == 0:
            continue
        if last_nonzero and value != last_nonzero:
            event = float(grid[index])
            if not events or event - events[-1] >= REVERSAL_MIN_SEPARATION_SEC:
                events.append(event)
        last_nonzero = int(value)
    return events


def reversal_metrics(
    run: ReplayRun,
    trace: FollowupTrace,
    *,
    derived_sample_dt_sec: float,
    derived_window: int = PRIMARY_DERIVED_WINDOW,
) -> dict[str, float | int]:
    events = commanded_reversal_times(
        run, sample_dt_sec=derived_sample_dt_sec, window_length=derived_window
    )
    grid, real_q, real_dq, valid = uniform_derived_velocity(
        run.t, run.q,
        sample_dt_sec=derived_sample_dt_sec,
        window_length=derived_window,
    )
    _, sim_q, sim_dq, sim_valid = uniform_derived_velocity(
        trace.t, trace.encoder_q,
        sample_dt_sec=derived_sample_dt_sec,
        window_length=derived_window,
    )
    valid &= sim_valid[:len(valid)]
    real_current = np.interp(grid, run.t, run.current)
    sim_current = np.interp(grid, trace.t, trace.current)
    local = np.zeros(len(grid), dtype=bool)
    for event in events:
        local |= np.abs(grid - event) <= REVERSAL_HALF_WINDOW_SEC
    local &= valid
    if not np.any(local):
        return {
            "event_count": len(events),
            "sample_count": 0,
            "position_mae": math.nan,
            "position_rmse": math.nan,
            "derived_velocity_mae": math.nan,
            "derived_velocity_rmse": math.nan,
            "current_mae": math.nan,
            "current_rmse": math.nan,
        }
    result: dict[str, float | int] = {
        "event_count": len(events),
        "sample_count": int(np.count_nonzero(local)),
    }
    for name, predicted, measured in (
        ("position", sim_q[local], real_q[local]),
        ("derived_velocity", sim_dq[local], real_dq[local]),
        ("current", sim_current[local], real_current[local]),
    ):
        metric = _error_metrics(predicted, measured)
        result[f"{name}_mae"] = metric["mae"]
        result[f"{name}_rmse"] = metric["rmse"]
    return result


def _run_row(
    run: ReplayRun,
    trace: FollowupTrace,
    metrics: dict[str, float],
    *,
    derived_window: int,
) -> dict[str, Any]:
    return {
        "condition": run.metadata["condition"],
        "trajectory": run.metadata["trajectory"],
        "repeat": int(run.metadata["repeat"]),
        "source_attempt": str(run.attempt),
        "variant": trace.variant,
        "feedback_view": trace.feedback_view,
        "controller_dt_sec": trace.controller_dt_sec,
        "backlash_width_rad": trace.backlash_width_rad,
        "derived_window": derived_window,
        **metrics,
    }


def _finite_mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else math.nan


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)
    metric_names = [
        "position_mae", "position_rmse",
        "present_velocity_mae", "present_velocity_rmse",
        "derived_velocity_mae", "derived_velocity_rmse",
        "real_present_vs_derived_velocity_mae",
        "real_present_vs_derived_velocity_rmse",
        "pwm_mae", "pwm_rmse", "current_mae", "current_rmse",
        "maximum_abs_simulated_pwm_fraction",
        "maximum_abs_simulated_current_A",
        "maximum_backlash_limit_violation_rad",
        "maximum_backlash_limit_violation_fraction_of_half_width",
    ]
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        item = {name: value for name, value in zip(group_keys, key)}
        item["run_count"] = len(members)
        for metric in metric_names:
            if metric.startswith("maximum_"):
                item[metric] = float(max(float(row[metric]) for row in members))
            else:
                item[metric] = _finite_mean(row[metric] for row in members)
        output.append(item)
    return output


def aggregate_reversal_rows(
    rows: list[dict[str, Any]], group_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)
    output = []
    metrics = (
        "position_mae", "position_rmse", "derived_velocity_mae",
        "derived_velocity_rmse", "current_mae", "current_rmse",
    )
    for key, members in sorted(groups.items()):
        item = {name: value for name, value in zip(group_keys, key)}
        item["run_count"] = len(members)
        item["event_count"] = sum(int(row["event_count"]) for row in members)
        item["sample_count"] = sum(int(row["sample_count"]) for row in members)
        for metric in metrics:
            item[metric] = _finite_mean(row[metric] for row in members)
        output.append(item)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_run(
    path: Path,
    run: ReplayRun,
    traces: list[FollowupTrace],
    *,
    derived_sample_dt_sec: float,
    derived_window: int,
    title_prefix: str,
) -> None:
    fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=False)
    axes[0].plot(run.t, run.goal, "k--", label="goal")
    axes[0].plot(run.t, run.q, label="real")
    axes[1].plot(run.t, run.dq, label="real Present Velocity")
    grid, _, real_derived, valid = uniform_derived_velocity(
        run.t, run.q,
        sample_dt_sec=derived_sample_dt_sec,
        window_length=derived_window,
    )
    axes[2].plot(grid[valid], real_derived[valid], label="real derived")
    axes[3].plot(run.t, run.pwm, label="real")
    axes[4].plot(run.t, run.current, label="real")
    duplicate_variants = len({trace.variant for trace in traces}) != len(traces)
    for trace in traces:
        label = trace.variant.replace("m3_backlash_", "backlash ")
        if duplicate_variants:
            label = f"{label} controller_dt={trace.controller_dt_sec:.3f}s"
        axes[0].plot(run.t, _sample(trace, trace.encoder_q, run.t), label=label)
        axes[1].plot(run.t, _sample(trace, trace.encoder_dq, run.t), label=label)
        _, _, derived, derived_valid = uniform_derived_velocity(
            trace.t, trace.encoder_q,
            sample_dt_sec=derived_sample_dt_sec,
            window_length=derived_window,
        )
        use = valid & derived_valid[:len(valid)]
        axes[2].plot(grid[use], derived[use], label=label)
        axes[3].plot(run.t, _sample(trace, trace.duty, run.t), label=label)
        axes[4].plot(run.t, _sample(trace, trace.current, run.t), label=label)
        axes[5].plot(trace.t, trace.backlash_q, label=label)
        axes[6].plot(trace.t, trace.output_q, label=f"output {label}")
    labels = (
        "position [rad]", "Present/qvel [rad/s]",
        f"derived velocity W={derived_window} [rad/s]", "PWM duty", "current [A]",
        "passive backlash angle [rad]", "physical output angle [rad]",
    )
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", ncol=3)
    axes[-1].set_xlabel("run-local time [s]")
    fig.suptitle(
        f"{title_prefix} — {run.metadata['condition']} / {run.metadata['trajectory']} / repeat 3"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _percentage_change(candidate: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else math.inf
    return 100.0 * (candidate - baseline) / baseline


def _width_variant(feedback_view: str, width_scale: float) -> str:
    label = f"{width_scale:.3f}".replace(".", "p")
    return f"m3_backlash_{feedback_view}_w{label}"


def _screen_backlash(
    aggregate: list[dict[str, Any]],
    primary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_variant = {row["variant"]: row for row in aggregate}
    baseline = by_variant["m3"]
    candidates = []
    for variant in sorted(name for name in by_variant if name != "m3"):
        candidate = by_variant[variant]
        changes = {
            metric: _percentage_change(candidate[metric], baseline[metric])
            for metric in (
                "position_rmse", "derived_velocity_rmse", "pwm_rmse", "current_rmse"
            )
        }
        paired = {}
        for metric in ("position_rmse", "derived_velocity_rmse", "pwm_rmse", "current_rmse"):
            wins = 0
            total = 0
            for base_run in (row for row in primary_rows if row["variant"] == "m3"):
                match = next(row for row in primary_rows if (
                    row["variant"] == variant
                    and row["condition"] == base_run["condition"]
                    and row["trajectory"] == base_run["trajectory"]
                ))
                wins += int(float(match[metric]) < float(base_run[metric]))
                total += 1
            paired[metric] = {"wins": wins, "total": total}
        constraint_ok = (
            candidate["maximum_backlash_limit_violation_fraction_of_half_width"]
            <= MAXIMUM_BACKLASH_LIMIT_VIOLATION_FRACTION
        )
        qualifies = (
            constraint_ok
            and
            changes["position_rmse"] <= 2.0
            and changes["derived_velocity_rmse"] <= 2.0
            and changes["pwm_rmse"] <= 10.0
            and changes["current_rmse"] <= 10.0
            and (
                changes["position_rmse"] <= -2.0
                or changes["derived_velocity_rmse"] <= -2.0
            )
            and (
                paired["position_rmse"]["wins"] >= 15
                or paired["derived_velocity_rmse"]["wins"] >= 15
            )
        )
        candidates.append({
            "variant": variant,
            "backlash_width_rad": float(candidate["backlash_width_rad"]),
            "maximum_limit_violation_rad": float(
                candidate["maximum_backlash_limit_violation_rad"]
            ),
            "maximum_limit_violation_fraction_of_half_width": float(
                candidate["maximum_backlash_limit_violation_fraction_of_half_width"]
            ),
            "constraint_quality_passes": constraint_ok,
            "percent_change_vs_m3_negative_is_better": changes,
            "paired_run_wins": paired,
            "passes_predeclared_screen": qualifies,
        })
    passing = [row["variant"] for row in candidates if row["passes_predeclared_screen"]]
    decision = "adopt_candidate" if len(passing) == 1 else (
        "additional_review_required" if len(passing) > 1 else "do_not_adopt"
    )
    return {
        "decision": decision,
        "passing_variants": passing,
        "screen_is_not_parameter_fitting": True,
        "thresholds": {
            "position_and_derived_velocity_max_degradation_percent": 2.0,
            "pwm_and_current_max_degradation_percent": 10.0,
            "minimum_primary_improvement_percent": 2.0,
            "minimum_paired_run_wins_out_of_24": 15,
            "maximum_limit_violation_fraction_of_half_width": (
                MAXIMUM_BACKLASH_LIMIT_VIOLATION_FRACTION
            ),
        },
        "candidates": candidates,
    }


def _compact_anomalies(rows: list[dict[str, Any]], count: int = 5) -> list[dict[str, Any]]:
    baseline = [row for row in rows if row["variant"] == "m3"]
    worst = sorted(baseline, key=lambda row: float(row["position_rmse"]), reverse=True)[:count]
    return [{
        "condition": row["condition"],
        "trajectory": row["trajectory"],
        "position_rmse": float(row["position_rmse"]),
        "derived_velocity_rmse": float(row["derived_velocity_rmse"]),
        "current_rmse": float(row["current_rmse"]),
    } for row in worst]


def run_all(args: argparse.Namespace) -> Path:
    cfg = load_campaign(args.config, require_bench=True)
    if args.repeat not in cfg.validation_repetitions:
        raise ValueError(
            f"repeat {args.repeat} is not held-out validation; expected {cfg.validation_repetitions}"
        )
    if args.canonical_controller_dt not in args.controller_dt:
        raise ValueError("canonical controller dt must be included in controller dt candidates")
    if PRIMARY_DERIVED_WINDOW not in args.derived_windows:
        raise ValueError(f"derived windows must include {PRIMARY_DERIVED_WINDOW}")
    if 0.0 not in args.backlash_width_scales:
        raise ValueError("backlash width scales must include the zero-width baseline")
    if any(scale < 0.0 for scale in args.backlash_width_scales):
        raise ValueError("backlash width scales cannot be negative")
    if len(set(args.backlash_width_scales)) != len(args.backlash_width_scales):
        raise ValueError("backlash width scales must be unique")
    if args.backlash_physics_timestep > args.canonical_controller_dt:
        raise ValueError("backlash physics timestep cannot exceed controller timestep")
    parameters, controller = load_identified_model(cfg, "m3")
    backlash_path = cfg.results_root / str(cfg.campaign_id) / "stage_backlash.yaml"
    if not backlash_path.exists():
        raise FileNotFoundError(f"effective backlash result is missing: {backlash_path}")
    backlash_result = yaml.safe_load(backlash_path.read_text())
    backlash_width = float(backlash_result["effective_backlash_width_rad"])

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    destination = args.output or (
        cfg.results_root / str(cfg.campaign_id) / "followup_validation" / stamp
    )
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "plots" / "cadence").mkdir(parents=True)
    (destination / "plots" / "backlash").mkdir(parents=True)

    runs = [
        load_replay_run(cfg, condition, trajectory, args.repeat)
        for condition in LOADED_CONDITIONS
        for trajectory in LOADED_TRAJECTORIES
    ]
    positive_width_scales = sorted(
        scale for scale in args.backlash_width_scales if scale > 0.0
    )
    total_units = len(runs) * (
        len(args.controller_dt) + 1 + 2 * len(positive_width_scales)
    )
    completed = 0
    all_rows: list[dict[str, Any]] = []
    reversal_rows: list[dict[str, Any]] = []
    primary_traces: dict[tuple[str, str], list[FollowupTrace]] = {}

    for run in runs:
        cadence_traces = []
        for controller_dt in args.controller_dt:
            trace = simulate_followup(
                cfg, run, parameters, controller,
                controller_dt_sec=controller_dt,
                physics_timestep_sec=args.physics_timestep,
            )
            cadence_traces.append(trace)
            for window in args.derived_windows:
                metrics = trace_metrics(
                    run, trace,
                    derived_sample_dt_sec=args.derived_sample_dt,
                    derived_window=window,
                )
                row = _run_row(run, trace, metrics, derived_window=window)
                row["experiment"] = "controller_cadence"
                all_rows.append(row)
            completed += 1
            print(
                f"[{completed:03d}/{total_units}] cadence dt={controller_dt:.3f}s "
                f"{run.metadata['condition']}/{run.metadata['trajectory']}",
                flush=True,
            )
        _plot_run(
            destination / "plots" / "cadence"
            / f"{run.metadata['condition']}__{run.metadata['trajectory']}.png",
            run, cadence_traces,
            derived_sample_dt_sec=args.derived_sample_dt,
            derived_window=PRIMARY_DERIVED_WINDOW,
            title_prefix="controller cadence sensitivity",
        )
        # Recompute the zero-width baseline at the same finer physics timestep
        # as every backlash candidate.  This prevents integrator resolution
        # from being attributed to backlash.
        baseline = simulate_followup(
            cfg, run, parameters, controller,
            controller_dt_sec=args.canonical_controller_dt,
            physics_timestep_sec=args.backlash_physics_timestep,
        )
        comparison_traces = [baseline]
        for window in args.derived_windows:
            metrics = trace_metrics(
                run, baseline,
                derived_sample_dt_sec=args.derived_sample_dt,
                derived_window=window,
            )
            row = _run_row(run, baseline, metrics, derived_window=window)
            row["experiment"] = "backlash_sensitivity"
            row["backlash_width_scale"] = 0.0
            all_rows.append(row)
        completed += 1
        print(
            f"[{completed:03d}/{total_units}] backlash baseline width=0x "
            f"{run.metadata['condition']}/{run.metadata['trajectory']}",
            flush=True,
        )
        for width_scale in positive_width_scales:
            candidate_width = backlash_width * width_scale
            for feedback in ("actuator_side", "output_side"):
                trace = simulate_followup(
                    cfg, run, parameters, controller,
                    controller_dt_sec=args.canonical_controller_dt,
                    physics_timestep_sec=args.backlash_physics_timestep,
                    backlash_width_rad=candidate_width,
                    feedback_view=feedback,
                )
                trace = FollowupTrace(
                    **{
                        **trace.__dict__,
                        "variant": _width_variant(feedback, width_scale),
                    }
                )
                comparison_traces.append(trace)
                for window in args.derived_windows:
                    metrics = trace_metrics(
                        run, trace,
                        derived_sample_dt_sec=args.derived_sample_dt,
                        derived_window=window,
                    )
                    row = _run_row(run, trace, metrics, derived_window=window)
                    row["experiment"] = "backlash_sensitivity"
                    row["backlash_width_scale"] = width_scale
                    all_rows.append(row)
                completed += 1
                print(
                    f"[{completed:03d}/{total_units}] backlash {feedback} "
                    f"width={width_scale:g}x "
                    f"{run.metadata['condition']}/{run.metadata['trajectory']}",
                    flush=True,
                )
        primary_traces[(run.metadata["condition"], run.metadata["trajectory"])] = comparison_traces
        for trace in comparison_traces:
            local = reversal_metrics(
                run, trace,
                derived_sample_dt_sec=args.derived_sample_dt,
            )
            reversal_rows.append({
                "condition": run.metadata["condition"],
                "trajectory": run.metadata["trajectory"],
                "variant": trace.variant,
                "feedback_view": trace.feedback_view,
                "backlash_width_rad": trace.backlash_width_rad,
                **local,
            })
        _plot_run(
            destination / "plots" / "backlash"
            / f"{run.metadata['condition']}__{run.metadata['trajectory']}.png",
            run, comparison_traces,
            derived_sample_dt_sec=args.derived_sample_dt,
            derived_window=PRIMARY_DERIVED_WINDOW,
            title_prefix="M3 versus stateful backlash",
        )

    primary_rows = [
            row for row in all_rows
            if row["derived_window"] == PRIMARY_DERIVED_WINDOW
            and row["experiment"] == "backlash_sensitivity"
    ]
    cadence_primary = [
        row for row in all_rows
        if row["experiment"] == "controller_cadence"
        and row["derived_window"] == PRIMARY_DERIVED_WINDOW
    ]
    backlash_primary = primary_rows

    cadence_overall = aggregate_rows(cadence_primary, group_keys=("controller_dt_sec",))
    cadence_trajectory = aggregate_rows(
        cadence_primary, group_keys=("controller_dt_sec", "trajectory")
    )
    baseline_sensitivity = aggregate_rows(
        [row for row in all_rows if (
            row["experiment"] == "controller_cadence"
            and math.isclose(row["controller_dt_sec"], args.canonical_controller_dt)
        )],
        group_keys=("derived_window",),
    )
    backlash_overall = aggregate_rows(
        backlash_primary,
        group_keys=("variant", "backlash_width_scale", "backlash_width_rad"),
    )
    backlash_trajectory = aggregate_rows(
        backlash_primary,
        group_keys=("variant", "backlash_width_scale", "trajectory"),
    )
    backlash_condition = aggregate_rows(
        backlash_primary,
        group_keys=("variant", "backlash_width_scale", "condition"),
    )
    reversal_overall = aggregate_reversal_rows(reversal_rows, ("variant",))
    reversal_trajectory = aggregate_reversal_rows(
        reversal_rows, ("variant", "trajectory")
    )
    screening = _screen_backlash(backlash_overall, backlash_primary)

    _write_csv(destination / "all_run_metrics.csv", all_rows)
    _write_csv(destination / "reversal_metrics.csv", reversal_rows)
    _write_csv(destination / "cadence_overall.csv", cadence_overall)
    _write_csv(destination / "backlash_overall.csv", backlash_overall)
    report = {
        "schema_version": 1,
        "campaign_id": cfg.campaign_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "contract": {
            "raw_data_modified": False,
            "parameters_refit": False,
            "model_parameters_source": str(
                cfg.results_root / str(cfg.campaign_id) / "stage_m3.yaml"
            ),
            "controller_source": str(
                cfg.results_root / str(cfg.campaign_id) / "stage_1_time_controller.yaml"
            ),
            "validation_repeat": args.repeat,
            "run_count": len(runs),
        },
        "settings": {
            "physics_timestep_sec": args.physics_timestep,
            "backlash_physics_timestep_sec": args.backlash_physics_timestep,
            "controller_dt_candidates_sec": args.controller_dt,
            "canonical_controller_dt_sec": args.canonical_controller_dt,
            "derived_sample_dt_sec": args.derived_sample_dt,
            "derived_windows": args.derived_windows,
            "primary_derived_window": PRIMARY_DERIVED_WINDOW,
            "effective_backlash_total_width_rad": backlash_width,
            "backlash_width_scales": args.backlash_width_scales,
            "maximum_allowed_limit_violation_fraction_of_half_width": (
                MAXIMUM_BACKLASH_LIMIT_VIOLATION_FRACTION
            ),
        },
        "controller_cadence": {
            "overall": cadence_overall,
            "by_trajectory": cadence_trajectory,
        },
        "derived_velocity_window_sensitivity": baseline_sensitivity,
        "backlash_ab": {
            "overall": backlash_overall,
            "by_trajectory": backlash_trajectory,
            "by_condition": backlash_condition,
            "screening": screening,
        },
        "reversal_local_100ms": {
            "overall": reversal_overall,
            "by_trajectory": reversal_trajectory,
        },
    }
    (destination / "full_summary.yaml").write_text(
        yaml.safe_dump(report, sort_keys=False)
    )
    compact = {
        "campaign_id": cfg.campaign_id,
        "result_directory": str(destination),
        "validation_repeat": args.repeat,
        "run_count": len(runs),
        "no_refit": True,
        "controller_cadence_overall": cadence_overall,
        "derived_velocity_window_sensitivity": baseline_sensitivity,
        "backlash_overall": backlash_overall,
        "reversal_overall": reversal_overall,
        "backlash_screening": screening,
        "worst_five_baseline_position_runs": _compact_anomalies(backlash_primary),
        "review_files": {
            "full_summary": str(destination / "full_summary.yaml"),
            "all_run_metrics": str(destination / "all_run_metrics.csv"),
            "reversal_metrics": str(destination / "reversal_metrics.csv"),
            "plots": str(destination / "plots"),
        },
    }
    (destination / "codex_summary.yaml").write_text(
        yaml.safe_dump(compact, sort_keys=False)
    )
    lines = [
        "Mode-3 M3 follow-up validation",
        f"campaign: {cfg.campaign_id}",
        f"held-out runs: {len(runs)} (repeat {args.repeat})",
        "parameters refit: no",
        "",
        "Controller cadence (macro mean across 24 runs):",
    ]
    for row in cadence_overall:
        lines.append(
            f"  dt={row['controller_dt_sec']:.3f}s | position RMSE={row['position_rmse']:.7f} rad "
            f"| derived velocity RMSE={row['derived_velocity_rmse']:.5f} rad/s "
            f"| PWM RMSE={row['pwm_rmse']:.5f} | current RMSE={row['current_rmse']:.5f} A"
        )
    lines.extend(("", "Backlash width sensitivity (canonical cadence, W=7):"))
    for row in backlash_overall:
        lines.append(
            f"  {row['variant']} width={row['backlash_width_scale']:g}x "
            f"| position RMSE={row['position_rmse']:.7f} rad "
            f"| derived velocity RMSE={row['derived_velocity_rmse']:.5f} rad/s "
            f"| PWM RMSE={row['pwm_rmse']:.5f} | current RMSE={row['current_rmse']:.5f} A "
            f"| max limit violation={100.0 * row['maximum_backlash_limit_violation_fraction_of_half_width']:.2f}% half-width"
        )
    lines.extend((
        "",
        f"Predeclared backlash screen: {screening['decision']}",
        "This screen is an A/B validation aid, not parameter fitting or proof of physical backlash.",
        "Give codex_summary.yaml to Codex first; inspect full CSV/plots only when needed.",
    ))
    (destination / "REPORT.txt").write_text("\n".join(lines) + "\n")
    latest = destination.parent / "LATEST.txt"
    latest.write_text(str(destination) + "\n")
    print(f"COMPLETE: {destination}", flush=True)
    print(f"CODEX SUMMARY: {destination / 'codex_summary.yaml'}", flush=True)
    return destination


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all existing-data M3 follow-up validations without refitting."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--controller-dt", type=float, nargs="+", default=[0.001, 0.010],
        help="equivalent controller update intervals to compare",
    )
    parser.add_argument(
        "--canonical-controller-dt", type=float, default=0.001,
        help="fixed cadence used for backlash sensitivity (default: 1 ms)",
    )
    parser.add_argument("--physics-timestep", type=float, default=0.001)
    parser.add_argument(
        "--backlash-physics-timestep", type=float, default=0.0001,
        help="finer timestep used by both zero-width baseline and backlash candidates",
    )
    parser.add_argument(
        "--backlash-width-scales", type=float, nargs="+",
        default=[0.0, 0.5, 1.0, 1.5],
        help="multipliers of the identified effective total backlash width",
    )
    parser.add_argument("--derived-sample-dt", type=float, default=0.010)
    parser.add_argument(
        "--derived-windows", type=int, nargs="+", default=[5, 7, 9]
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    run_all(argument_parser().parse_args())


if __name__ == "__main__":
    main()
