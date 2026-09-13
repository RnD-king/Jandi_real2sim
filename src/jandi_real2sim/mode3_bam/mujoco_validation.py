"""MuJoCo simulation and replay validation for Mode-3 BAM actuator models.

The default path builds commands from the canonical trajectory YAML and saves
MuJoCo-only traces.  Measured telemetry is loaded only when comparison with an
explicit repeat is requested.  No parameter is optimized here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
import yaml

from .acquisition import current_raw_to_joint_a
from .config import (
    DEFAULT_CAMPAIGN,
    DISTANCE_KEYS,
    LOADED_TRAJECTORIES,
    MASS_KEYS,
    Campaign,
    load_campaign,
    trajectory_profile_matches,
)
from .trajectories import build


SUPPORTED_MODELS = ("m1", "m3")
LOADED_CONDITIONS = tuple(
    f"{mass}_{distance}" for mass in MASS_KEYS for distance in DISTANCE_KEYS
)


@dataclass(frozen=True)
class ReplayRun:
    attempt: Path
    metadata: dict[str, Any]
    t: np.ndarray
    goal: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    pwm: np.ndarray
    current: np.ndarray
    voltage: np.ndarray
    torque_enable: np.ndarray


@dataclass(frozen=True)
class RigProperties:
    physical_mass_kg: float
    center_of_mass_m: float
    inertia_about_pivot_kg_m2: float
    inertia_about_com_kg_m2: float
    load_radius_m: float


@dataclass(frozen=True)
class SimulationTrace:
    model_name: str
    t: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    duty: np.ndarray
    current: np.ndarray
    motor_torque: np.ndarray
    friction_torque: np.ndarray
    friction_limit: np.ndarray


def _condition_metadata(cfg: Campaign, condition: str, trajectory: str) -> dict[str, Any]:
    selected = cfg.condition(condition)
    return {
        "campaign_id": cfg.campaign_id,
        "condition": condition,
        "trajectory": trajectory,
        "load_disk_count": selected.disk_count,
        "load_mass_kg": selected.mass_kg,
        "load_intrinsic_inertia_kg_m2": selected.intrinsic_inertia_kg_m2,
        "load_disk_diameter_m": cfg.bench["load"]["disk_diameter_m"],
        "axis_to_load_com_m": selected.distance_m,
        "arm_mass_kg": cfg.bench["arm"]["mass_kg"],
        "arm_com_radius_m": cfg.bench["arm"]["com_radius_m"],
        "arm_inertia_kg_m2": cfg.bench["arm"]["inertia_about_pivot_kg_m2"],
    }


def build_simulation_run(
    cfg: Campaign,
    condition: str,
    trajectory: str,
    *,
    supply_voltage_v: float = 12.0,
) -> ReplayRun:
    """Build a MuJoCo-only input without reading measured telemetry."""
    if condition not in LOADED_CONDITIONS:
        raise ValueError(f"MuJoCo loaded condition must be one of {LOADED_CONDITIONS}")
    if trajectory not in LOADED_TRAJECTORIES:
        raise ValueError(f"trajectory must be one of {LOADED_TRAJECTORIES}")
    if not math.isfinite(supply_voltage_v) or supply_voltage_v <= 0.0:
        raise ValueError("supply voltage must be positive and finite")
    samples = build(cfg, trajectory)
    t = np.asarray([sample.time_sec for sample in samples], dtype=float)
    goal = np.asarray([sample.goal_rad for sample in samples], dtype=float)
    torque_enable = np.asarray(
        [sample.torque_enable for sample in samples], dtype=bool
    )
    zeros = np.zeros_like(t)
    metadata = _condition_metadata(cfg, condition, trajectory)
    metadata.update({
        "source": "generated_from_trajectory_yaml",
        "supply_voltage_V": supply_voltage_v,
    })
    return ReplayRun(
        attempt=Path("[generated: no measured repeat]"),
        metadata=metadata,
        t=t,
        goal=goal,
        q=np.full_like(t, goal[0]),
        dq=zeros.copy(),
        pwm=zeros.copy(),
        current=zeros.copy(),
        voltage=np.full_like(t, supply_voltage_v),
        torque_enable=torque_enable,
    )


def _latest_valid_attempt(
    cfg: Campaign, condition: str, trajectory: str, repeat: int
) -> Path:
    logical = (
        cfg.output_root
        / str(cfg.campaign_id)
        / condition
        / trajectory
        / f"repeat_{repeat}"
    )
    valid: list[Path] = []
    for attempt in sorted(logical.glob("attempt_*")):
        metadata_path = attempt / "metadata.json"
        telemetry_path = attempt / "telemetry.csv"
        if not metadata_path.exists() or not telemetry_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            metadata.get("valid") is True
            and metadata.get("condition") == condition
            and metadata.get("trajectory") == trajectory
            and int(metadata.get("repeat", -1)) == repeat
            and trajectory_profile_matches(cfg, trajectory, metadata)
        ):
            valid.append(attempt)
    if not valid:
        raise FileNotFoundError(
            f"valid replay log not found: {condition}/{trajectory}/repeat_{repeat}"
        )
    return valid[-1]


def load_replay_run(
    cfg: Campaign, condition: str, trajectory: str, repeat: int
) -> ReplayRun:
    if condition not in LOADED_CONDITIONS:
        raise ValueError(f"MuJoCo loaded condition must be one of {LOADED_CONDITIONS}")
    if trajectory not in LOADED_TRAJECTORIES:
        raise ValueError(f"trajectory must be one of {LOADED_TRAJECTORIES}")
    attempt = _latest_valid_attempt(cfg, condition, trajectory, repeat)
    metadata = json.loads((attempt / "metadata.json").read_text())
    with (attempt / "telemetry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError(f"telemetry has fewer than two samples: {attempt}")

    def array(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=float)

    raw_t = array("host_time_sec")
    if "present_current_raw" in rows[0]:
        recorded_directions = metadata.get("signal_directions", {})
        current_direction = int(
            recorded_directions.get("current", cfg.hardware["current_direction"])
        )
        measured_current = np.asarray(
            [current_raw_to_joint_a(
                cfg, int(float(row["present_current_raw"])),
                direction=current_direction,
            )
             for row in rows],
            dtype=float,
        )
        metadata["comparison_current_source"] = "present_current_raw"
        metadata["comparison_current_direction"] = current_direction
    else:
        measured_current = array("present_current_A")
        metadata["comparison_current_source"] = "legacy_present_current_A"
    return ReplayRun(
        attempt=attempt,
        metadata=metadata,
        t=raw_t - raw_t[0],
        goal=array("goal_position_rad"),
        q=array("present_position_rad"),
        dq=array("present_velocity_rad_s"),
        pwm=array("present_pwm_fraction"),
        current=measured_current,
        voltage=array("input_voltage_V"),
        torque_enable=array("torque_enable") > 0.5,
    )


def load_identified_model(
    cfg: Campaign, model_name: str
) -> tuple[dict[str, float], dict[str, float]]:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"model must be one of {SUPPORTED_MODELS}")
    root = cfg.results_root / str(cfg.campaign_id)
    stage_path = root / f"stage_{model_name}.yaml"
    controller_path = root / "stage_1_time_controller.yaml"
    if not stage_path.exists():
        raise FileNotFoundError(f"identified model is missing: {stage_path}")
    if not controller_path.exists():
        raise FileNotFoundError(f"controller result is missing: {controller_path}")
    stage = yaml.safe_load(stage_path.read_text())
    controller = yaml.safe_load(controller_path.read_text())
    parameters = {key: float(value) for key, value in stage["parameters"].items()}
    controller_values = {
        key: float(value)
        for key, value in controller.items()
        if key in (
            "command_delay_sec",
            "pwm_error_gain_per_raw_rad",
            "pwm_velocity_gain_s_per_rad",
            "position_p_gain",
        )
    }
    return parameters, controller_values


def rig_properties(metadata: dict[str, Any]) -> RigProperties:
    load_mass = float(metadata["load_mass_kg"])
    load_radius = float(metadata["axis_to_load_com_m"])
    arm_mass = float(metadata["arm_mass_kg"])
    arm_radius = float(metadata["arm_com_radius_m"])
    total_mass = load_mass + arm_mass
    if total_mass <= 0.0:
        raise ValueError("loaded MuJoCo bench requires positive physical mass")
    center = (load_mass * load_radius + arm_mass * arm_radius) / total_mass
    pivot_inertia = (
        float(metadata["arm_inertia_kg_m2"])
        + load_mass * load_radius * load_radius
        + float(metadata.get("load_intrinsic_inertia_kg_m2", 0.0))
    )
    com_inertia = pivot_inertia - total_mass * center * center
    if com_inertia <= 0.0:
        raise ValueError(
            "aggregated bench inertia about COM is not positive; check mass/COM/inertia metadata"
        )
    return RigProperties(
        physical_mass_kg=total_mass,
        center_of_mass_m=center,
        inertia_about_pivot_kg_m2=pivot_inertia,
        inertia_about_com_kg_m2=com_inertia,
        load_radius_m=load_radius,
    )


def bench_xml(
    cfg: Campaign,
    run: ReplayRun,
    armature_kg_m2: float,
    physics_timestep_sec: float,
) -> str:
    rig = rig_properties(run.metadata)
    if physics_timestep_sec <= 0.0:
        raise ValueError("physics timestep must be positive")
    gravity_sign = float(cfg.bench["gravity_torque_sign"])
    gravity_zero = float(cfg.bench["gravity_zero_angle_rad"])
    if gravity_sign not in (-1.0, 1.0):
        raise ValueError("gravity_torque_sign must be -1 or +1")
    if not math.isclose(gravity_zero, 0.0, abs_tol=1e-12):
        raise ValueError("MuJoCo bench currently requires gravity_zero_angle_rad=0")
    inertia = rig.inertia_about_com_kg_m2
    visual_length = max(rig.load_radius_m, 2.0 * rig.center_of_mass_m, 0.08)
    disk_radius = 0.5 * float(run.metadata.get("load_disk_diameter_m", 0.07))
    axis_y = gravity_sign
    return f"""
<mujoco model="jandi_mode3_bam_bench">
  <compiler angle="radian" inertiafromgeom="false"/>
  <option timestep="{physics_timestep_sec:.12g}" gravity="0 0 -{float(cfg.bench['gravity_m_s2']):.12g}" integrator="implicitfast"/>
  <visual><global azimuth="135" elevation="-15"/></visual>
  <worldbody>
    <light pos="0 -1 1.5" dir="0 1 -1"/>
    <body name="pendulum" pos="0 0 0">
      <joint name="pivot" type="hinge" axis="0 {axis_y:.0f} 0" armature="{armature_kg_m2:.12g}" damping="0" frictionloss="0"/>
      <inertial pos="0 0 {rig.center_of_mass_m:.12g}" mass="{rig.physical_mass_kg:.12g}"
                diaginertia="{inertia:.12g} {inertia:.12g} {inertia:.12g}"/>
      <geom name="arm_visual" type="capsule" fromto="0 0 0 0 0 {visual_length:.12g}"
            size="0.006" rgba="0.35 0.45 0.55 1" contype="0" conaffinity="0"/>
      <geom name="load_visual" type="cylinder" pos="0 0 {rig.load_radius_m:.12g}"
            size="{disk_radius:.12g} 0.012" rgba="0.85 0.55 0.12 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def _zoh(t: np.ndarray, values: np.ndarray, query: float) -> float:
    index = int(np.searchsorted(t, query, side="right") - 1)
    return float(values[min(max(index, 0), len(values) - 1)])


def _friction_limit(
    model_name: str,
    parameters: dict[str, float],
    dq: float,
    motor_torque: float,
    external_torque: float,
) -> float:
    value = (
        parameters["coulomb_friction_nm"]
        + parameters["viscous_friction_nm_s_per_rad"] * abs(dq)
    )
    if model_name == "m3":
        value += parameters["load_friction"] * abs(motor_torque - external_torque)
    return max(0.0, value)


def simulate_mujoco(
    cfg: Campaign,
    run: ReplayRun,
    model_name: str,
    parameters: dict[str, float],
    controller: dict[str, float],
    *,
    physics_timestep_sec: float = 0.001,
    viewer: bool = False,
) -> SimulationTrace:
    xml = bench_xml(
        cfg, run, parameters["armature_kg_m2"], physics_timestep_sec
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[0] = run.q[0]
    data.qvel[0] = run.dq[0]
    mujoco.mj_forward(model, data)

    duration = float(run.t[-1])
    count = int(math.ceil(duration / physics_timestep_sec)) + 1
    sample_t = np.minimum(np.arange(count, dtype=float) * physics_timestep_sec, duration)
    q = np.empty(count); dq = np.empty(count); duty = np.empty(count)
    current = np.empty(count); motor_torque = np.empty(count)
    friction_torque = np.empty(count); friction_limit = np.empty(count)
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

    viewer_context = None
    if viewer:
        import mujoco.viewer as mj_viewer
        viewer_context = mj_viewer.launch_passive(model, data)
    wall_start = time.monotonic()
    try:
        for index, now in enumerate(sample_t):
            q_now = float(data.qpos[0]); dq_now = float(data.qvel[0])
            goal = _zoh(run.t, run.goal, now - delay)
            voltage = float(np.interp(now, run.t, run.voltage))
            enabled = _zoh(run.t, run.torque_enable.astype(float), now) > 0.5
            command = np.clip(
                controller["pwm_error_gain_per_raw_rad"] * p_gain * (goal - q_now)
                - controller["pwm_velocity_gain_s_per_rad"] * dq_now,
                -pwm_limit,
                pwm_limit,
            )
            if enabled:
                predicted_current = (
                    command * voltage - parameters["kt_nm_per_a"] * dq_now
                ) / parameters["resistance_ohm"]
                motor = parameters["kt_nm_per_a"] * predicted_current
            else:
                command = 0.0
                predicted_current = 0.0
                motor = 0.0
            external = (
                float(cfg.bench["gravity_torque_sign"])
                * gravity_scale
                * math.sin(q_now - float(cfg.bench["gravity_zero_angle_rad"]))
            )
            limit = _friction_limit(
                model_name, parameters, dq_now, motor, external
            )
            model.dof_frictionloss[0] = limit
            data.qfrc_applied[0] = motor
            mujoco.mj_forward(model, data)

            q[index] = q_now; dq[index] = dq_now; duty[index] = command
            current[index] = predicted_current; motor_torque[index] = motor
            friction_torque[index] = float(data.qfrc_constraint[0])
            friction_limit[index] = limit

            if index + 1 < count:
                mujoco.mj_step(model, data)
                if viewer_context is not None:
                    viewer_context.sync()
                    remaining = wall_start + float(sample_t[index + 1]) - time.monotonic()
                    if remaining > 0.0:
                        time.sleep(remaining)
    finally:
        if viewer_context is not None:
            viewer_context.close()
    return SimulationTrace(
        model_name=model_name,
        t=sample_t,
        q=q,
        dq=dq,
        duty=duty,
        current=current,
        motor_torque=motor_torque,
        friction_torque=friction_torque,
        friction_limit=friction_limit,
    )


def _at_log_times(run: ReplayRun, trace: SimulationTrace, values: np.ndarray) -> np.ndarray:
    return np.interp(run.t, trace.t, values)


def trace_metrics(run: ReplayRun, trace: SimulationTrace) -> dict[str, float]:
    predicted = {
        "position": _at_log_times(run, trace, trace.q),
        "velocity": _at_log_times(run, trace, trace.dq),
        "pwm": _at_log_times(run, trace, trace.duty),
        "current": _at_log_times(run, trace, trace.current),
    }
    measured = {
        "position": run.q,
        "velocity": run.dq,
        "pwm": run.pwm,
        "current": run.current,
    }
    result: dict[str, float] = {}
    for name in predicted:
        error = predicted[name] - measured[name]
        result[f"{name}_mae"] = float(np.mean(np.abs(error)))
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error * error)))
    result["maximum_abs_simulated_pwm_fraction"] = float(np.max(np.abs(predicted["pwm"])))
    result["maximum_abs_simulated_current_A"] = float(np.max(np.abs(predicted["current"])))
    return result


def _write_comparison_csv(
    path: Path, run: ReplayRun, traces: Iterable[SimulationTrace]
) -> None:
    traces = tuple(traces)
    fields = [
        "time_sec", "goal_position_rad", "input_voltage_V", "torque_enable",
        "real_position_rad", "real_velocity_rad_s", "real_pwm_fraction", "real_current_A",
    ]
    for trace in traces:
        prefix = trace.model_name
        fields.extend((
            f"{prefix}_position_rad", f"{prefix}_velocity_rad_s",
            f"{prefix}_pwm_fraction", f"{prefix}_current_A",
            f"{prefix}_motor_torque_Nm", f"{prefix}_friction_torque_Nm",
            f"{prefix}_friction_limit_Nm",
        ))
    sampled: dict[str, dict[str, np.ndarray]] = {}
    for trace in traces:
        sampled[trace.model_name] = {
            "q": _at_log_times(run, trace, trace.q),
            "dq": _at_log_times(run, trace, trace.dq),
            "pwm": _at_log_times(run, trace, trace.duty),
            "current": _at_log_times(run, trace, trace.current),
            "motor": _at_log_times(run, trace, trace.motor_torque),
            "friction": _at_log_times(run, trace, trace.friction_torque),
            "limit": _at_log_times(run, trace, trace.friction_limit),
        }
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, now in enumerate(run.t):
            row: dict[str, float | int] = {
                "time_sec": float(now),
                "goal_position_rad": float(run.goal[index]),
                "input_voltage_V": float(run.voltage[index]),
                "torque_enable": int(run.torque_enable[index]),
                "real_position_rad": float(run.q[index]),
                "real_velocity_rad_s": float(run.dq[index]),
                "real_pwm_fraction": float(run.pwm[index]),
                "real_current_A": float(run.current[index]),
            }
            for trace in traces:
                values = sampled[trace.model_name]
                prefix = trace.model_name
                row.update({
                    f"{prefix}_position_rad": float(values["q"][index]),
                    f"{prefix}_velocity_rad_s": float(values["dq"][index]),
                    f"{prefix}_pwm_fraction": float(values["pwm"][index]),
                    f"{prefix}_current_A": float(values["current"][index]),
                    f"{prefix}_motor_torque_Nm": float(values["motor"][index]),
                    f"{prefix}_friction_torque_Nm": float(values["friction"][index]),
                    f"{prefix}_friction_limit_Nm": float(values["limit"][index]),
                })
            writer.writerow(row)


def _write_simulation_csv(
    path: Path, run: ReplayRun, traces: Iterable[SimulationTrace]
) -> None:
    traces = tuple(traces)
    fields = ["time_sec", "goal_position_rad", "input_voltage_V", "torque_enable"]
    for trace in traces:
        prefix = trace.model_name
        fields.extend((
            f"{prefix}_position_rad", f"{prefix}_velocity_rad_s",
            f"{prefix}_pwm_fraction", f"{prefix}_current_A",
            f"{prefix}_motor_torque_Nm", f"{prefix}_friction_torque_Nm",
            f"{prefix}_friction_limit_Nm",
        ))
    reference = traces[0]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, now in enumerate(reference.t):
            row: dict[str, float | int] = {
                "time_sec": float(now),
                "goal_position_rad": _zoh(run.t, run.goal, float(now)),
                "input_voltage_V": float(np.interp(now, run.t, run.voltage)),
                "torque_enable": int(
                    _zoh(run.t, run.torque_enable.astype(float), float(now)) > 0.5
                ),
            }
            for trace in traces:
                prefix = trace.model_name
                row.update({
                    f"{prefix}_position_rad": float(trace.q[index]),
                    f"{prefix}_velocity_rad_s": float(trace.dq[index]),
                    f"{prefix}_pwm_fraction": float(trace.duty[index]),
                    f"{prefix}_current_A": float(trace.current[index]),
                    f"{prefix}_motor_torque_Nm": float(trace.motor_torque[index]),
                    f"{prefix}_friction_torque_Nm": float(trace.friction_torque[index]),
                    f"{prefix}_friction_limit_Nm": float(trace.friction_limit[index]),
                })
            writer.writerow(row)


def _save_plot(path: Path, run: ReplayRun, traces: Iterable[SimulationTrace]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces = tuple(traces)
    fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    axes[0].plot(run.t, run.goal, "k--", label="q_cmd")
    axes[0].plot(run.t, run.q, label="real")
    axes[1].plot(run.t, run.dq, label="real")
    axes[2].plot(run.t, run.pwm, label="real")
    axes[3].plot(run.t, run.current, label="real")
    for trace in traces:
        axes[0].plot(run.t, _at_log_times(run, trace, trace.q), label=f"MuJoCo {trace.model_name.upper()}")
        axes[1].plot(run.t, _at_log_times(run, trace, trace.dq), label=f"MuJoCo {trace.model_name.upper()}")
        axes[2].plot(run.t, _at_log_times(run, trace, trace.duty), label=f"MuJoCo {trace.model_name.upper()}")
        axes[3].plot(run.t, _at_log_times(run, trace, trace.current), label=f"MuJoCo {trace.model_name.upper()}")
        axes[4].plot(run.t, _at_log_times(run, trace, trace.motor_torque), label=f"motor {trace.model_name.upper()}")
        axes[4].plot(run.t, _at_log_times(run, trace, trace.friction_torque), "--", label=f"friction {trace.model_name.upper()}")
    labels = (
        "angle [rad]", "velocity [rad/s]", "PWM duty", "current [A]", "predicted torque [Nm]",
    )
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label); axis.grid(True, alpha=0.3); axis.legend(loc="best", ncol=3)
    axes[-1].set_xlabel("run-local time [s]")
    fig.suptitle(
        f"{run.metadata['condition']} / {run.metadata['trajectory']} / repeat {run.metadata['repeat']}"
    )
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _save_simulation_plot(
    path: Path, run: ReplayRun, traces: Iterable[SimulationTrace]
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces = tuple(traces)
    fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    axes[0].plot(run.t, run.goal, "k--", label="q_cmd")
    for trace in traces:
        label = f"MuJoCo {trace.model_name.upper()}"
        axes[0].plot(trace.t, trace.q, label=label)
        axes[1].plot(trace.t, trace.dq, label=label)
        axes[2].plot(trace.t, trace.duty, label=label)
        axes[3].plot(trace.t, trace.current, label=label)
        axes[4].plot(trace.t, trace.motor_torque, label=f"motor {trace.model_name.upper()}")
        axes[4].plot(
            trace.t, trace.friction_torque, "--",
            label=f"friction {trace.model_name.upper()}",
        )
    labels = (
        "angle [rad]", "velocity [rad/s]", "PWM duty", "current [A]",
        "predicted torque [Nm]",
    )
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", ncol=3)
    axes[-1].set_xlabel("simulation time [s]")
    fig.suptitle(
        f"MuJoCo only — {run.metadata['condition']} / {run.metadata['trajectory']}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_simulation(
    cfg: Campaign,
    condition: str,
    trajectory: str,
    model_names: Iterable[str],
    *,
    supply_voltage_v: float = 12.0,
    physics_timestep_sec: float = 0.001,
    viewer: bool = False,
    output: Path | None = None,
) -> Path:
    """Run one YAML trajectory without loading or comparing measured data."""
    model_names = tuple(model_names)
    if not model_names or any(name not in SUPPORTED_MODELS for name in model_names):
        raise ValueError(f"models must be selected from {SUPPORTED_MODELS}")
    if viewer and len(model_names) != 1:
        raise ValueError("--viewer requires exactly one model: m1 or m3")
    run = build_simulation_run(
        cfg, condition, trajectory, supply_voltage_v=supply_voltage_v
    )
    traces: list[SimulationTrace] = []
    parameter_sets: dict[str, dict[str, float]] = {}
    for model_name in model_names:
        parameters, controller = load_identified_model(cfg, model_name)
        parameter_sets[model_name] = parameters
        traces.append(simulate_mujoco(
            cfg, run, model_name, parameters, controller,
            physics_timestep_sec=physics_timestep_sec,
            viewer=viewer,
        ))
    destination = output or (
        cfg.results_root / str(cfg.campaign_id) / "mujoco_simulation"
        / condition / trajectory
    )
    destination.mkdir(parents=True, exist_ok=True)
    _write_simulation_csv(destination / "simulation.csv", run, traces)
    _save_simulation_plot(destination / "simulation.png", run, traces)
    rig = rig_properties(run.metadata)
    report = {
        "campaign_id": cfg.campaign_id,
        "source": "generated_from_trajectory_yaml",
        "condition": condition,
        "trajectory": trajectory,
        "comparison_repeat": None,
        "supply_voltage_V": supply_voltage_v,
        "physics_timestep_sec": physics_timestep_sec,
        "bench": {
            "physical_mass_kg": rig.physical_mass_kg,
            "center_of_mass_m": rig.center_of_mass_m,
            "physical_inertia_about_pivot_kg_m2": rig.inertia_about_pivot_kg_m2,
            "physical_inertia_about_com_kg_m2": rig.inertia_about_com_kg_m2,
        },
        "models": {
            trace.model_name: {
                "parameters": parameter_sets[trace.model_name],
                "maximum_abs_simulated_pwm_fraction": float(
                    np.max(np.abs(trace.duty))
                ),
                "maximum_abs_simulated_current_A": float(
                    np.max(np.abs(trace.current))
                ),
            }
            for trace in traces
        },
    }
    (destination / "summary.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    return destination


def validate_replay(
    cfg: Campaign,
    condition: str,
    trajectory: str,
    repeat: int,
    model_names: Iterable[str],
    *,
    physics_timestep_sec: float = 0.001,
    viewer: bool = False,
    output: Path | None = None,
) -> Path:
    model_names = tuple(model_names)
    if not model_names or any(name not in SUPPORTED_MODELS for name in model_names):
        raise ValueError(f"models must be selected from {SUPPORTED_MODELS}")
    if viewer and len(model_names) != 1:
        raise ValueError("--viewer requires exactly one model: m1 or m3")
    run = load_replay_run(cfg, condition, trajectory, repeat)
    traces: list[SimulationTrace] = []
    parameter_sets: dict[str, dict[str, float]] = {}
    for model_name in model_names:
        parameters, controller = load_identified_model(cfg, model_name)
        parameter_sets[model_name] = parameters
        traces.append(simulate_mujoco(
            cfg, run, model_name, parameters, controller,
            physics_timestep_sec=physics_timestep_sec,
            viewer=viewer,
        ))
    destination = output or (
        cfg.results_root / str(cfg.campaign_id) / "mujoco_validation"
        / condition / trajectory / f"repeat_{repeat}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    _write_comparison_csv(destination / "comparison.csv", run, traces)
    _save_plot(destination / "comparison.png", run, traces)
    rig = rig_properties(run.metadata)
    report = {
        "campaign_id": cfg.campaign_id,
        "source_attempt": str(run.attempt),
        "condition": condition,
        "trajectory": trajectory,
        "repeat": repeat,
        "validation_only": repeat in cfg.validation_repetitions,
        "measured_current_source": run.metadata["comparison_current_source"],
        "measured_current_direction": run.metadata.get(
            "comparison_current_direction"
        ),
        "physics_timestep_sec": physics_timestep_sec,
        "bench": {
            "physical_mass_kg": rig.physical_mass_kg,
            "center_of_mass_m": rig.center_of_mass_m,
            "physical_inertia_about_pivot_kg_m2": rig.inertia_about_pivot_kg_m2,
            "physical_inertia_about_com_kg_m2": rig.inertia_about_com_kg_m2,
        },
        "models": {
            trace.model_name: {
                "parameters": parameter_sets[trace.model_name],
                "metrics": trace_metrics(run, trace),
            }
            for trace in traces
        },
    }
    (destination / "summary.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    return destination


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Mode-3 BAM trajectories in MuJoCo. By default no measured data "
            "is loaded; --compare-repeat enables measured replay comparison."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--condition", required=True, choices=LOADED_CONDITIONS)
    parser.add_argument(
        "--trajectory", choices=LOADED_TRAJECTORIES,
        help="run only this trajectory; omitted means all four loaded trajectories",
    )
    parser.add_argument(
        "--compare-repeat", type=int, choices=(1, 2, 3),
        help="compare against this measured repeat; omitted means MuJoCo-only",
    )
    parser.add_argument("--model", choices=("m1", "m3", "both"), default="both")
    parser.add_argument(
        "--voltage", type=float, default=12.0,
        help="constant supply voltage for MuJoCo-only runs (default: 12.0 V)",
    )
    parser.add_argument("--physics-timestep", type=float, default=0.001)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    parser = argument_parser()
    args = parser.parse_args()
    cfg = load_campaign(args.config, require_bench=True)
    models = ("m1", "m3") if args.model == "both" else (args.model,)
    trajectories = (
        (args.trajectory,) if args.trajectory is not None else LOADED_TRAJECTORIES
    )
    multiple = len(trajectories) > 1
    for trajectory in trajectories:
        destination_override = None
        if args.output is not None:
            destination_override = args.output / trajectory if multiple else args.output
        if args.compare_repeat is None:
            destination = run_simulation(
                cfg, args.condition, trajectory, models,
                supply_voltage_v=args.voltage,
                physics_timestep_sec=args.physics_timestep,
                viewer=args.viewer,
                output=destination_override,
            )
            print(f"MuJoCo simulation: {destination}")
        else:
            destination = validate_replay(
                cfg, args.condition, trajectory, args.compare_repeat, models,
                physics_timestep_sec=args.physics_timestep,
                viewer=args.viewer,
                output=destination_override,
            )
            summary = yaml.safe_load((destination / "summary.yaml").read_text())
            print(f"MuJoCo comparison: {destination}")
            for name, result in summary["models"].items():
                metrics = result["metrics"]
                print(
                    f"{name.upper()}: position MAE={metrics['position_mae']:.8f} rad, "
                    f"RMSE={metrics['position_rmse']:.8f} rad, "
                    f"current RMSE={metrics['current_rmse']:.6f} A"
                )


if __name__ == "__main__":
    main()
