"""Ordered BAM-style fitting stages. Repeats 1/2 fit; repeat 3 never fits."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from scipy.optimize import least_squares

from .config import Campaign, LOADED_TRAJECTORIES, trajectory_profile_matches


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class Run:
    metadata: dict[str, Any]
    t: np.ndarray
    goal: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    pwm: np.ndarray
    voltage: np.ndarray
    torque_enable: np.ndarray


MODEL_PARAMETERS = {
    "m1": ("kt_nm_per_a", "resistance_ohm", "armature_kg_m2", "coulomb_friction_nm", "viscous_friction_nm_s_per_rad"),
    "m2": ("kt_nm_per_a", "resistance_ohm", "armature_kg_m2", "coulomb_friction_nm", "viscous_friction_nm_s_per_rad",
           "stribeck_friction_nm", "stribeck_velocity_rad_s", "stribeck_alpha"),
    "m3": ("kt_nm_per_a", "resistance_ohm", "armature_kg_m2", "coulomb_friction_nm", "viscous_friction_nm_s_per_rad",
           "load_friction"),
    "m4": ("kt_nm_per_a", "resistance_ohm", "armature_kg_m2", "coulomb_friction_nm", "viscous_friction_nm_s_per_rad",
           "stribeck_friction_nm", "stribeck_velocity_rad_s", "stribeck_alpha", "load_friction", "load_stribeck_friction"),
    "m5": ("kt_nm_per_a", "resistance_ohm", "armature_kg_m2", "coulomb_friction_nm", "viscous_friction_nm_s_per_rad",
           "stribeck_friction_nm", "stribeck_velocity_rad_s", "stribeck_alpha",
           "motor_load_friction", "external_load_friction",
           "motor_load_stribeck_friction", "external_load_stribeck_friction"),
}


def result_root(cfg: Campaign) -> Path:
    if not cfg.campaign_id:
        raise ValueError("campaign.id를 먼저 지정하십시오.")
    root = cfg.results_root / str(cfg.campaign_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _selected_attempt(cfg: Campaign, logical: Path, trajectory: str) -> Path | None:
    valid = []
    for path in sorted(logical.glob("attempt_*")):
        metadata = path / "metadata.json"
        if metadata.exists():
            report = json.loads(metadata.read_text())
            if (report.get("valid") and
                    trajectory_profile_matches(cfg, trajectory, report)):
                valid.append(path)
    return valid[-1] if valid else None


def load_runs(cfg: Campaign, *, condition: str | None = None,
              repetitions: tuple[int, ...] | None = None) -> list[Run]:
    root = cfg.output_root / str(cfg.campaign_id)
    runs = []
    for metadata_path in root.glob("*/*/repeat_*/attempt_*/metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        repeat = int(metadata["repeat"])
        if not metadata.get("valid") or (condition and metadata["condition"] != condition): continue
        if not trajectory_profile_matches(cfg, metadata["trajectory"], metadata): continue
        if repetitions is not None and repeat not in repetitions: continue
        logical = metadata_path.parents[1]
        if _selected_attempt(cfg, logical, metadata["trajectory"]) != metadata_path.parent: continue
        with (metadata_path.parent / "telemetry.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        def array(name: str) -> np.ndarray: return np.asarray([float(row[name]) for row in rows], dtype=float)
        runs.append(Run(metadata, array("host_time_sec"), array("goal_position_rad"),
                        array("present_position_rad"), array("present_velocity_rad_s"),
                        array("present_pwm_fraction"), array("input_voltage_V"),
                        array("torque_enable") > 0.5))
    return sorted(runs, key=lambda run: (run.metadata["condition"], run.metadata["trajectory"], run.metadata["repeat"]))


def _zoh(source_t: np.ndarray, source_values: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """Previous-value hold for discrete Goal Position commands."""
    indexes = np.searchsorted(source_t, query_t, side="right") - 1
    return source_values[np.clip(indexes, 0, len(source_values) - 1)]


def fit_time_controller(cfg: Campaign, progress: ProgressCallback | None = None) -> Path:
    runs = [run for run in load_runs(cfg, condition="no_load", repetitions=cfg.fit_repetitions)
            if run.metadata["trajectory"] == "delay_probe"]
    if not runs: raise ValueError("repeat 1/2의 valid no_load/delay_probe가 없습니다.")
    lo, hi = map(float, cfg.fit["bounds"]["command_delay_sec"])
    delays = np.linspace(lo, hi, 101)
    best = None
    p_gain = 850.0
    started = time.monotonic()
    for index, delay in enumerate(delays, start=1):
        design, observed = [], []
        for run in runs:
            delayed = _zoh(run.t, run.goal, run.t - delay)
            mask = run.torque_enable & (np.abs(run.pwm) < .95 * float(cfg.safety["maximum_abs_pwm_fraction"]))
            error = delayed - run.q
            design.append(np.column_stack((p_gain * error[mask], -run.dq[mask])))
            observed.append(run.pwm[mask])
        x = np.linalg.lstsq(np.vstack(design), np.concatenate(observed), rcond=None)[0]
        residual = np.vstack(design) @ x - np.concatenate(observed)
        score = float(np.mean(residual ** 2))
        if best is None or score < best[0]: best = (score, delay, x)
        if progress is not None and (index == 1 or index % 5 == 0 or index == len(delays)):
            progress({
                "stage": "fit_time", "evaluation": index, "total": len(delays),
                "elapsed_sec": time.monotonic() - started,
                "current_rmse": math.sqrt(score), "best_rmse": math.sqrt(best[0]),
                "command_delay_sec": float(delay),
            })
    assert best is not None
    payload = {
        "stage": "time_controller", "created_at": datetime.now().astimezone().isoformat(),
        "fit_repetitions": list(cfg.fit_repetitions), "validation_repetitions": list(cfg.validation_repetitions),
        "command_delay_sec": float(best[1]), "pwm_error_gain_per_raw_rad": float(best[2][0]),
        "pwm_velocity_gain_s_per_rad": float(best[2][1]), "pwm_fit_mse": float(best[0]),
        "position_p_gain": 850,
    }
    path = result_root(cfg) / "stage_1_time_controller.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False)); return path


def _controller(cfg: Campaign) -> dict[str, float]:
    path = result_root(cfg) / "stage_1_time_controller.yaml"
    if not path.exists(): raise FileNotFoundError("먼저 '1. 시간/제어기 특성' 피팅을 실행하십시오.")
    return yaml.safe_load(path.read_text())


def _initial(cfg: Campaign, model: str) -> dict[str, float]:
    initial = {key: float(value) for key, value in cfg.fit["initial"].items()}
    if model == "m1": return initial
    parents = (("m1",) if model in ("m2", "m3") else
               (("m2", "m3") if model == "m4" else ("m4",)))
    for parent in parents:
        path = result_root(cfg) / f"stage_{parent}.yaml"
        if not path.exists(): raise FileNotFoundError(f"먼저 {parent.upper()} 피팅을 실행하십시오.")
        initial.update({key: float(value) for key, value in yaml.safe_load(path.read_text())["parameters"].items()})
    if model == "m5":
        # M5 splits M4's load coefficients into motor-side and external-side
        # directional coefficients.  Equal values reproduce the M4 starting point.
        initial["motor_load_friction"] = initial["load_friction"]
        initial["external_load_friction"] = initial["load_friction"]
        initial["motor_load_stribeck_friction"] = initial["load_stribeck_friction"]
        initial["external_load_stribeck_friction"] = initial["load_stribeck_friction"]
    return initial


def _friction_budget(model: str, p: dict[str, float], dq: float,
                     motor_torque: float, external_torque: float) -> float:
    budget = p["coulomb_friction_nm"] + p["viscous_friction_nm_s_per_rad"] * abs(dq)
    load = abs(motor_torque - external_torque)
    if model in ("m3", "m4"): budget += p["load_friction"] * load
    if model == "m5":
        budget += abs(p["motor_load_friction"] * motor_torque
                      - p["external_load_friction"] * external_torque)
    if model in ("m2", "m4", "m5"):
        factor = math.exp(-abs(dq / p["stribeck_velocity_rad_s"]) ** p["stribeck_alpha"])
        budget += factor * p["stribeck_friction_nm"]
        if model == "m4": budget += factor * p["load_stribeck_friction"] * load
        elif model == "m5":
            budget += factor * abs(p["motor_load_stribeck_friction"] * motor_torque
                                   - p["external_load_stribeck_friction"] * external_torque)
    return budget


def _system_inertia(armature_kg_m2: float, metadata: dict[str, Any]) -> float:
    """Pivot inertia including the arm and finite-size disk load assembly."""

    mass = float(metadata["load_mass_kg"])
    radius = float(metadata["axis_to_load_com_m"])
    return (armature_kg_m2 + float(metadata["arm_inertia_kg_m2"])
            + mass * radius * radius
            + float(metadata.get("load_intrinsic_inertia_kg_m2", 0.0)))


def simulate(cfg: Campaign, run: Run, model: str, p: dict[str, float], controller: dict[str, float]) -> np.ndarray:
    dt = float(cfg.fit["resample_dt_sec"])
    t = np.arange(0.0, run.t[-1] + dt * .5, dt)
    goal = np.interp(t - float(controller["command_delay_sec"]), run.t, run.goal,
                     left=run.goal[0], right=run.goal[-1])
    vin = np.interp(t, run.t, run.voltage); enabled = np.interp(t, run.t, run.torque_enable.astype(float)) > .5
    condition_mass = float(run.metadata["load_mass_kg"]); radius = float(run.metadata["axis_to_load_com_m"])
    arm_mass = float(run.metadata["arm_mass_kg"]); arm_com = float(run.metadata["arm_com_radius_m"])
    inertia = _system_inertia(p["armature_kg_m2"], run.metadata)
    gravity_scale = condition_mass * float(cfg.bench["gravity_m_s2"]) * radius + arm_mass * float(cfg.bench["gravity_m_s2"]) * arm_com
    gravity_sign = float(cfg.bench["gravity_torque_sign"]); zero = float(cfg.bench["gravity_zero_angle_rad"])
    pwm_limit = int(cfg.registers["expected_pwm_limit_raw"]) * 0.00113
    q = float(np.interp(0.0, run.t, run.q)); dq = float(np.interp(0.0, run.t, run.dq)); output = np.empty_like(t)
    for index in range(len(t)):
        output[index] = q
        error = goal[index] - q
        duty = np.clip(controller["pwm_error_gain_per_raw_rad"] * 850.0 * error
                       - controller["pwm_velocity_gain_s_per_rad"] * dq, -pwm_limit, pwm_limit)
        if enabled[index]:
            motor = p["kt_nm_per_a"] / p["resistance_ohm"] * duty * vin[index]
            motor -= p["kt_nm_per_a"] ** 2 / p["resistance_ohm"] * dq
        else: motor = 0.0
        external = gravity_sign * gravity_scale * math.sin(q - zero)
        budget = _friction_budget(model, p, dq, motor, external)
        without = motor + external
        stop = -(inertia / dt * dq + without)
        friction = float(np.clip(stop, -budget, budget))
        acceleration = (without + friction) / inertia
        q += dq * dt + .5 * acceleration * dt * dt; dq += acceleration * dt
    return np.interp(run.t, t, output)


def _metrics(cfg: Campaign, runs: list[Run], model: str, p: dict[str, float], controller: dict[str, float]) -> dict[str, float]:
    errors = [simulate(cfg, run, model, p, controller) - run.q for run in runs]
    values = np.concatenate(errors) if errors else np.asarray([math.nan])
    return {"position_mae_rad": float(np.mean(np.abs(values))), "position_rmse_rad": float(np.sqrt(np.mean(values ** 2))), "run_count": len(runs)}


def fit_model(cfg: Campaign, model: str,
              progress: ProgressCallback | None = None) -> Path:
    if model not in MODEL_PARAMETERS: raise KeyError(model)
    controller = _controller(cfg); initial = _initial(cfg, model); names = MODEL_PARAMETERS[model]
    runs = [run for run in load_runs(cfg, repetitions=cfg.fit_repetitions)
            if run.metadata["condition"] != "no_load" and run.metadata["trajectory"] in LOADED_TRAJECTORIES]
    validation = [run for run in load_runs(cfg, repetitions=cfg.validation_repetitions)
                  if run.metadata["condition"] != "no_load" and run.metadata["trajectory"] in LOADED_TRAJECTORIES]
    if not runs or not validation: raise ValueError("loaded repeat 1/2 fit와 repeat 3 validation 로그가 모두 필요합니다.")
    bounds = cfg.fit["bounds"]; lower = np.asarray([bounds[name][0] for name in names], float)
    upper = np.asarray([bounds[name][1] for name in names], float); x0 = np.asarray([initial[name] for name in names], float)
    stride = int(cfg.fit["evaluation_stride"])
    evaluations = 0
    started = time.monotonic()
    best_rmse = math.inf
    maximum = int(cfg.fit["maximum_function_evaluations"])
    maximum_residual_calls = maximum * (len(names) + 1)
    def residual(x: np.ndarray) -> np.ndarray:
        nonlocal evaluations, best_rmse
        p = dict(initial); p.update(dict(zip(names, map(float, x))))
        values = np.concatenate([(simulate(cfg, run, model, p, controller) - run.q)[::stride] for run in runs])
        evaluations += 1
        current_rmse = float(np.sqrt(np.mean(values ** 2)))
        best_rmse = min(best_rmse, current_rmse)
        if progress is not None and (evaluations == 1 or evaluations % 5 == 0):
            progress({
                "stage": f"fit_{model}", "model": model,
                "evaluation": evaluations, "total": maximum_residual_calls,
                "elapsed_sec": time.monotonic() - started,
                "current_rmse": current_rmse, "best_rmse": best_rmse,
                "parameters": dict(zip(names, map(float, x))),
            })
        return values
    fit = least_squares(residual, np.clip(x0, lower, upper), bounds=(lower, upper),
                        max_nfev=maximum, verbose=0)
    if progress is not None:
        progress({
            "stage": f"fit_{model}", "model": model,
            "evaluation": evaluations, "total": maximum_residual_calls,
            "elapsed_sec": time.monotonic() - started,
            "current_rmse": float(np.sqrt(np.mean(fit.fun ** 2))),
            "best_rmse": best_rmse, "parameters": dict(zip(names, map(float, fit.x))),
            "finished": True,
        })
    parameters = dict(initial); parameters.update(dict(zip(names, map(float, fit.x))))
    payload = {
        "stage": model, "created_at": datetime.now().astimezone().isoformat(), "model": model,
        "parameters": {name: parameters[name] for name in names}, "fit": _metrics(cfg, runs, model, parameters, controller),
        "validation": _metrics(cfg, validation, model, parameters, controller),
        "optimizer": {"success": bool(fit.success), "message": fit.message, "evaluations": int(fit.nfev)},
        "fit_repetitions": list(cfg.fit_repetitions), "validation_repetitions": list(cfg.validation_repetitions),
    }
    path = result_root(cfg) / f"stage_{model}.yaml"; path.write_text(yaml.safe_dump(payload, sort_keys=False)); return path


def compare_models(cfg: Campaign) -> Path:
    candidates = []
    for model in MODEL_PARAMETERS:
        path = result_root(cfg) / f"stage_{model}.yaml"
        if not path.exists(): raise FileNotFoundError(f"{model.upper()} 결과가 없습니다.")
        payload = yaml.safe_load(path.read_text()); candidates.append(payload)
    threshold = float(cfg.fit["validation_improvement_threshold"])
    selected = candidates[0]
    for candidate in candidates[1:]:
        if candidate["validation"]["position_mae_rad"] < selected["validation"]["position_mae_rad"] * (1.0 - threshold):
            selected = candidate
    report = {
        "selected_model": selected["model"], "selection_rule": f"simpler retained unless validation MAE improves by >={threshold:.1%}",
        "models": {item["model"]: {"fit": item["fit"], "validation": item["validation"]} for item in candidates},
        "parameters": selected["parameters"],
    }
    root = result_root(cfg)
    path = root / "selected_model.yaml"; path.write_text(yaml.safe_dump(report, sort_keys=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [item["model"].upper() for item in candidates]
    fit_mae = [item["fit"]["position_mae_rad"] for item in candidates]
    validation_mae = [item["validation"]["position_mae_rad"] for item in candidates]
    x = np.arange(len(names)); width = .36
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(x-width/2, fit_mae, width, label="repeat 1·2 fit")
    axis.bar(x+width/2, validation_mae, width, label="repeat 3 validation")
    axis.set_xticks(x, names); axis.set_ylabel("position MAE [rad]"); axis.grid(axis="y", alpha=.25); axis.legend()
    axis.set_title(f"Selected: {selected['model'].upper()}")
    fig.tight_layout(); fig.savefig(root / "model_comparison.png", dpi=160); plt.close(fig)
    return path


def fit_backlash(cfg: Campaign) -> Path:
    selected = result_root(cfg) / "selected_model.yaml"
    if not selected.exists(): raise FileNotFoundError("먼저 M1~M5 비교/선택을 실행하십시오.")
    runs = [run for run in load_runs(cfg, condition="no_load", repetitions=cfg.fit_repetitions)
            if run.metadata["trajectory"] == "backlash_probe"]
    if not runs: raise ValueError("repeat 1/2의 backlash_probe가 없습니다.")
    positive, negative = [], []
    for run in runs:
        error = run.goal - run.q
        positive.extend(error[run.dq > 0.01]); negative.extend(error[run.dq < -0.01])
    if not positive or not negative: raise ValueError("방향별 유효한 backlash 표본이 없습니다.")
    width = abs(float(np.median(positive)) - float(np.median(negative)))
    lo, hi = map(float, cfg.fit["bounds"]["backlash_width_rad"]); width = float(np.clip(width, lo, hi))
    payload = {"effective_backlash_width_rad": width,
               "warning": "Internal encoder trajectory에서 얻은 effective hysteresis이며 외부 출력축 계측 백래시와 동일하다고 보장하지 않습니다."}
    path = result_root(cfg) / "stage_backlash.yaml"; path.write_text(yaml.safe_dump(payload, sort_keys=False)); return path
