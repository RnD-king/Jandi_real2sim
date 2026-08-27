"""README-v3 staged identification, holdout validation, and reporting."""

from __future__ import annotations

import csv
import json
import math
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import least_squares

from .canonical_attempts import select_valid_attempt
from .canonical_config import CanonicalCampaign
from .canonical_model import replay, zoh_command
from .canonical_trajectories import dynamic_run_specs, static_run_specs
from .spec import DEFAULT_CAMPAIGN, DYNAMIC_RUN_COUNT, STATIC_RUN_COUNT


@dataclass(frozen=True)
class Run:
    path: Path
    metadata: dict[str, Any]
    columns: dict[str, np.ndarray]

    @property
    def mechanical(self) -> str:
        return str(self.metadata["mechanical_configuration"])

    @property
    def trajectory(self) -> str:
        return str(self.metadata["trajectory"])

    @property
    def repeat(self) -> int:
        return int(self.metadata["repeat"])


NUMERIC_COLUMNS = (
    "sample_index", "host_time_ns", "command_tx_after_ns", "command_event", "goal_position_rad",
    "present_position_rad", "present_velocity_rad_s", "present_current_A",
    "present_pwm_fraction", "temperature_C", "current_saturated", "pwm_saturated",
)


def _load_run(path: Path) -> Run:
    metadata = json.loads((path / "metadata.json").read_text())
    if not metadata.get("valid_flag"):
        raise ValueError(f"invalid raw run: {path}: {metadata.get('invalid_reason')}")
    with (path / "telemetry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty telemetry: {path}")
    columns: dict[str, np.ndarray] = {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in NUMERIC_COLUMNS
    }
    columns["phase"] = np.asarray([row["phase"] for row in rows], dtype=object)
    return Run(path, metadata, columns)


def _require_runs(cfg: CanonicalCampaign, kind: str) -> list[Run]:
    if cfg.campaign_id is None:
        raise ValueError("campaign.id가 미확정입니다.")
    root = cfg.output_root / cfg.campaign_id
    if kind == "static":
        paths = [root / spec.relative_directory for spec in static_run_specs(cfg)]
        expected = STATIC_RUN_COUNT
    elif kind == "dynamic":
        paths = [root / spec.relative_directory for spec in dynamic_run_specs(cfg)]
        expected = DYNAMIC_RUN_COUNT
    elif kind == "delay":
        paths = [root / "delay" / "probe_1"]
        expected = 1
    else:
        raise ValueError(kind)
    selected: list[Path] = []
    missing: list[Path] = []
    for logical in paths:
        try:
            attempt = select_valid_attempt(logical)
        except FileNotFoundError:
            missing.append(logical)
            continue
        if not (attempt / "metadata.json").is_file() or not (attempt / "telemetry.csv").is_file():
            missing.append(logical)
        else:
            selected.append(attempt)
    if missing:
        raise FileNotFoundError(f"canonical {kind} raw run 누락 ({len(missing)}/{expected}):\n" + "\n".join(map(str, missing)))
    runs = [_load_run(path) for path in selected]
    if len(runs) != expected:
        raise AssertionError(f"{kind} run count={len(runs)}, expected={expected}")
    return runs


def _robust_linear(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    initial = np.linalg.lstsq(x, y, rcond=None)[0]
    result = least_squares(lambda p: x @ p - y, initial, loss="soft_l1")
    residual = y - x @ result.x
    dof = max(1, len(y) - x.shape[1])
    variance = float(residual @ residual / dof)
    covariance = np.linalg.pinv(result.jac.T @ result.jac) * variance
    return result.x, np.sqrt(np.maximum(0.0, np.diag(covariance))), float(np.sqrt(np.mean(residual**2)))


def estimate_static(cfg: CanonicalCampaign, runs: list[Run], bootstrap: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, float | str]] = []
    rejected: dict[str, int] = {}
    rejected_total = 0
    gravity = float(cfg.geometry["gravity_m_s2"])
    offset = float(cfg.geometry["gravity_zero_angle_rad"])
    for run in runs:
        approach = "approach_positive" if "approach_positive" in str(run.path) else "approach_negative"
        branch = 1.0 if approach == "approach_positive" else -1.0
        c = run.columns
        for phase in dict.fromkeys(c["phase"].tolist()):
            if not str(phase).endswith("_averaging"):
                continue
            mask = c["phase"] == phase
            reasons = []
            if not np.any(mask):
                reasons.append("empty_averaging_window")
            else:
                if np.any(c["current_saturated"][mask] != 0):
                    reasons.append("current_saturation")
                if np.any(c["pwm_saturated"][mask] != 0):
                    reasons.append("pwm_saturation")
                if float(np.median(np.abs(c["present_velocity_rad_s"][mask]))) > float(cfg.trajectories["static_calibration"]["maximum_settled_abs_velocity_rad_s"]):
                    reasons.append("velocity_not_settled")
                if float(np.std(c["present_position_rad"][mask])) > float(cfg.trajectories["static_calibration"]["maximum_settled_position_std_rad"]):
                    reasons.append("position_not_stable")
                if float(np.std(c["present_current_A"][mask])) > float(cfg.trajectories["static_calibration"]["maximum_settled_current_std_A"]):
                    reasons.append("current_not_stable")
            if reasons:
                rejected_total += 1
                for reason in reasons:
                    rejected[reason] = rejected.get(reason, 0) + 1
                continue
            q = float(np.median(c["present_position_rad"][mask]))
            current = float(np.median(c["present_current_A"][mask]))
            goal = float(np.median(c["goal_position_rad"][mask]))
            moment = gravity * (
                float(cfg.geometry["arm_mass_kg"]) * float(cfg.geometry["arm_com_radius_m"])
                + cfg.load_mass_kg(run.mechanical) * cfg.arm_length_m(run.mechanical)
            )
            required_torque = float(cfg.geometry["gravity_torque_sign"]) * moment * math.sin(q + offset)
            rows.append({"q": q, "goal": goal, "error": goal - q, "current": current,
                         "required_torque": required_torque, "branch": branch,
                         "mechanical": run.mechanical, "approach": approach,
                         "run": str(run.path), "mass_kg": cfg.load_mass_kg(run.mechanical),
                         "arm_length_m": cfg.arm_length_m(run.mechanical)})
    if len(rows) < 12:
        raise ValueError(f"valid static plateau가 부족합니다: {len(rows)}")
    current = np.asarray([float(row["current"]) for row in rows])
    torque = np.asarray([float(row["required_torque"]) for row in rows])
    error = np.asarray([float(row["error"]) for row in rows])
    branch = np.asarray([float(row["branch"]) for row in rows])
    ktau, ktau_sigma, torque_rmse = _robust_linear(np.column_stack((current, branch, np.ones(len(rows)))), torque)
    ap, ap_sigma, current_rmse = _robust_linear(np.column_stack((error, branch, np.ones(len(rows)))), current)
    torque_design = np.column_stack((current, branch, np.ones(len(rows))))
    current_design = np.column_stack((error, branch, np.ones(len(rows))))
    repeat_count = int(bootstrap["repeat_count"])
    seed = int(bootstrap["random_seed"])
    if repeat_count < 2:
        raise ValueError("static bootstrap repeat_count는 2 이상이어야 합니다.")
    run_names = sorted(set(str(row["run"]) for row in rows))
    by_run = {name: [row for row in rows if row["run"] == name] for name in run_names}
    rng = np.random.default_rng(seed)
    bootstrap_ktau: list[float] = []
    bootstrap_ap: list[float] = []
    for _ in range(repeat_count):
        selected = rng.choice(run_names, size=len(run_names), replace=True)
        sampled = [row for name in selected for row in by_run[str(name)]]
        sample_current = np.asarray([float(row["current"]) for row in sampled])
        sample_torque = np.asarray([float(row["required_torque"]) for row in sampled])
        sample_error = np.asarray([float(row["error"]) for row in sampled])
        sample_branch = np.asarray([float(row["branch"]) for row in sampled])
        torque_x = np.column_stack((sample_current, sample_branch, np.ones(len(sampled))))
        current_x = np.column_stack((sample_error, sample_branch, np.ones(len(sampled))))
        if np.linalg.matrix_rank(torque_x) < 3 or np.linalg.matrix_rank(current_x) < 3:
            continue
        bootstrap_ktau.append(float(_robust_linear(torque_x, sample_torque)[0][0]))
        bootstrap_ap.append(float(_robust_linear(current_x, sample_current)[0][0]))
    if len(bootstrap_ktau) < max(2, repeat_count // 2):
        raise ValueError("run-level static bootstrap의 유효 표본이 부족합니다.")
    ktau_bootstrap_std = float(np.std(bootstrap_ktau, ddof=1))
    ap_bootstrap_std = float(np.std(bootstrap_ap, ddof=1))
    ktau_ci = [float(value) for value in np.percentile(bootstrap_ktau, [2.5, 97.5])]
    ap_ci = [float(value) for value in np.percentile(bootstrap_ap, [2.5, 97.5])]
    torque_condition = float(np.linalg.cond(torque_design))
    current_condition = float(np.linalg.cond(current_design))
    warning_threshold = float(bootstrap["condition_number_warning_threshold"])
    warnings = []
    if torque_condition > warning_threshold:
        warnings.append("Ktau regression design is ill-conditioned")
    if current_condition > warning_threshold:
        warnings.append("aP regression design is ill-conditioned")
    for index, row in enumerate(rows):
        row["torque_residual_Nm"] = float(torque[index] - torque_design[index] @ ktau)
        row["current_residual_A"] = float(current[index] - current_design[index] @ ap)
    by_condition: dict[str, Any] = {}
    for mechanical in sorted(set(str(row["mechanical"]) for row in rows)):
        mask = np.asarray([row["mechanical"] == mechanical for row in rows])
        if np.count_nonzero(mask) >= 4:
            local, _, _ = _robust_linear(np.column_stack((error[mask], np.ones(np.count_nonzero(mask)))), current[mask])
            by_condition[mechanical] = {"aP_A_per_rad": float(local[0]), "points": int(np.count_nonzero(mask))}
    return {
        "model": "static_branch_robust_regression",
        "accepted_plateau_count": len(rows), "rejected_plateau_count": rejected_total,
        "reject_reason_counts": rejected,
        "Ktau_reference_from_stall_Nm_per_A": 1.615,
        "Ktau_eff_prior_Nm_per_A": float(ktau[0]),
        "Ktau_regression_sigma_Nm_per_A": float(ktau_sigma[0]),
        "Ktau_bootstrap_std_Nm_per_A": ktau_bootstrap_std, "Ktau_bootstrap_CI95_Nm_per_A": ktau_ci,
        "Ktau_static_uncertainty_Nm_per_A": max(float(ktau_sigma[0]), ktau_bootstrap_std),
        "Ktau_branch_difference_Nm": float(2.0 * abs(ktau[1])), "torque_intercept_Nm": float(ktau[2]),
        "aP_prior_A_per_rad": float(ap[0]), "aP_regression_sigma_A_per_rad": float(ap_sigma[0]),
        "aP_bootstrap_std_A_per_rad": ap_bootstrap_std, "aP_bootstrap_CI95_A_per_rad": ap_ci,
        "aP_uncertainty_A_per_rad": max(float(ap_sigma[0]), ap_bootstrap_std),
        "aP_branch_difference_A": float(2.0 * abs(ap[1])), "current_intercept_A": float(ap[2]),
        "torque_regression_rmse_Nm": torque_rmse, "current_regression_rmse_A": current_rmse,
        "regression_rank": {"Ktau": int(np.linalg.matrix_rank(torque_design)), "aP": int(np.linalg.matrix_rank(current_design))},
        "regression_condition_number": {"Ktau": torque_condition, "aP": current_condition},
        "bootstrap": {"unit": "whole_static_sweep_run", "repeat_count": repeat_count,
                      "valid_repeat_count": len(bootstrap_ktau), "random_seed": seed},
        "warnings": warnings, "diagnostic_rows": rows,
        "aP_by_mechanical_configuration": by_condition,
        "diagnostic_note": "possible load-dependent residual trend; inspect static_residual_diagnostics.png",
        "model_extension_decision": "manual review required; M1 is not extended automatically",
    }


def estimate_delay(cfg: CanonicalCampaign, run: Run) -> dict[str, Any]:
    c = run.columns
    t = c["host_time_ns"] * 1e-9
    event_mask = c["command_event"] == 1
    event_t = c["command_tx_after_ns"][event_mask] * 1e-9
    goal = c["goal_position_rad"][event_mask]
    current = c["present_current_A"]
    changes = np.flatnonzero(np.abs(np.diff(goal)) > 1e-9) + 1
    spec = cfg.trajectories["delay_probe"]
    threshold = float(spec["onset_current_threshold_A"])
    baseline_sec = float(spec["pre_event_baseline_sec"])
    search_sec = float(spec["response_search_sec"])
    rows = []
    for index in changes:
        command_time = event_t[index]
        before = (t >= command_time - baseline_sec) & (t < command_time)
        after = np.flatnonzero((t >= command_time) & (t <= command_time + search_sec))
        if not np.any(before) or not len(after):
            continue
        baseline = float(np.median(current[before]))
        onset = after[np.flatnonzero(np.abs(current[after] - baseline) >= threshold)]
        if not len(onset):
            continue
        amplitude = float(goal[index] - goal[index - 1])
        rows.append({"delay_s": float(t[onset[0]] - command_time), "direction": "positive" if amplitude > 0 else "negative", "amplitude_rad": abs(amplitude)})
    if not rows:
        raise ValueError("Present Current onset을 검출하지 못했습니다.")
    delays = np.asarray([row["delay_s"] for row in rows])
    by_direction = {name: float(np.mean([row["delay_s"] for row in rows if row["direction"] == name])) for name in ("positive", "negative") if any(row["direction"] == name for row in rows)}
    by_amplitude = {f"{amp:.9g}": float(np.mean([row["delay_s"] for row in rows if row["amplitude_rad"] == amp])) for amp in sorted(set(row["amplitude_rad"] for row in rows))}
    dt = float(np.median(np.diff(t)))
    reconstructed_goal = zoh_command(t, event_t, goal)
    goal_edge = np.diff(reconstructed_goal, prepend=reconstructed_goal[0])
    current_edge = np.diff(current, prepend=current[0])
    max_lag = max(1, round(search_sec / dt))
    scores = []
    for lag in range(max_lag + 1):
        left = goal_edge[:len(goal_edge) - lag or None]
        right = current_edge[lag:]
        scores.append(abs(float(np.dot(left, right))))
    alignment_delay = int(np.argmax(scores)) * dt
    measured_resolution = float(np.median(np.diff(t)))
    return {"method": "host_syncWriteTxOnly_to_present_current_threshold_plus_derivative_alignment", "delay_mean_s": float(np.mean(delays)),
            "delay_std_s": float(np.std(delays, ddof=1)) if len(delays) > 1 else 0.0,
            "delay_median_s": float(np.median(delays)), "delay_by_direction_s": by_direction,
            "delay_by_amplitude_s": by_amplitude, "sampling_resolution_s": measured_resolution,
            "requested_sampling_resolution_s": 1.0 / float(cfg.timing["delay_telemetry_target_rate_hz"] or cfg.command_rate_hz),
            "alignment_delay_s": float(alignment_delay), "event_count": len(rows), "events": rows}


def _load_fit_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 3:
        raise ValueError("fit.yaml schema_version은 3이어야 합니다.")
    required = {
        "physics_timestep_sec": raw.get("physics_timestep_sec"),
        "static_bootstrap.repeat_count": raw["static_bootstrap"].get("repeat_count"),
        "static_bootstrap.random_seed": raw["static_bootstrap"].get("random_seed"),
        "static_bootstrap.condition_number_warning_threshold": raw["static_bootstrap"].get("condition_number_warning_threshold"),
        "stage_d.parameter_bounds": raw["stage_d"].get("parameter_bounds"),
        "stage_d.initial_parameters": raw["stage_d"].get("initial_parameters"),
        "stage_d.independent_seeds": raw["stage_d"].get("independent_seeds"),
        "stage_d.max_function_evaluations": raw["stage_d"].get("max_function_evaluations"),
        "stage_d.evaluation_stride": raw["stage_d"].get("evaluation_stride"),
        "loss.position_scale_rad": raw["loss"].get("position_scale_rad"),
        "loss.velocity_scale_rad_s": raw["loss"].get("velocity_scale_rad_s"),
        "loss.current_scale_A": raw["loss"].get("current_scale_A"),
        "loss.weights": raw["loss"].get("weights"),
    }
    missing = [name for name, value in required.items() if value is None]
    if bool(raw["stage_e"].get("enabled")) and raw["stage_e"].get("uncertainty_bounds") is None:
        missing.append("stage_e.uncertainty_bounds")
    if missing:
        raise ValueError("fit 실행 전 미확정 항목:\n" + "\n".join(f"- {name}" for name in missing))
    return raw


def _dynamic_arrays(run: Run, stride: int) -> tuple[np.ndarray, ...]:
    c = run.columns
    t = (c["host_time_ns"] - c["host_time_ns"][0]) * 1e-9
    event = c["command_event"] == 1
    command_t = (c["command_tx_after_ns"][event] - c["host_time_ns"][0]) * 1e-9
    command_goal = c["goal_position_rad"][event]
    index = np.arange(0, len(t), stride)
    return (
        t[index], command_t, command_goal,
        c["present_position_rad"][index], c["present_velocity_rad_s"][index],
        c["present_current_A"][index], c["current_saturated"][index].astype(bool),
        c["pwm_saturated"][index].astype(bool),
    )


def primary_fit_sample_counts(prepared: list[tuple[Run, tuple[np.ndarray, ...]]]) -> dict[str, int]:
    """Describe Stage-D's validity region without hiding saturated diagnostics."""
    current = np.concatenate([arrays[6] for _, arrays in prepared]).astype(bool)
    pwm = np.concatenate([arrays[7] for _, arrays in prepared]).astype(bool)
    return {
        "total_sample_count": int(len(current)),
        "normal_sample_count": int(np.count_nonzero(~current & ~pwm)),
        "current_saturated_sample_count": int(np.count_nonzero(current)),
        "pwm_saturated_sample_count": int(np.count_nonzero(pwm)),
        # The current clip exists in M1.  The PWM/electrical limit does not.
        "excluded_from_primary_fit_count": int(np.count_nonzero(pwm)),
    }


def real_to_real_repeatability(runs: list[Run]) -> dict[str, Any]:
    """Time-align real repeats and compare all three pairs, independently of simulation."""
    groups: dict[tuple[str, str], list[Run]] = {}
    for run in runs:
        groups.setdefault((run.mechanical, run.trajectory), []).append(run)
    pair_rows: list[dict[str, Any]] = []
    for (mechanical, trajectory), grouped in sorted(groups.items()):
        by_repeat = {run.repeat: run for run in grouped}
        for left_repeat, right_repeat in ((1, 2), (1, 3), (2, 3)):
            if left_repeat not in by_repeat or right_repeat not in by_repeat:
                continue
            left, right = by_repeat[left_repeat], by_repeat[right_repeat]
            left_t = (left.columns["host_time_ns"] - left.columns["host_time_ns"][0]) * 1e-9
            right_t = (right.columns["host_time_ns"] - right.columns["host_time_ns"][0]) * 1e-9
            end = min(float(left_t[-1]), float(right_t[-1]))
            mask = left_t <= end
            common = left_t[mask]
            if len(common) < 2:
                continue
            row: dict[str, Any] = {
                "mechanical_configuration": mechanical, "trajectory": trajectory,
                "pair": f"r{left_repeat}-r{right_repeat}", "sample_count": int(len(common)),
            }
            for column, metric in (
                ("present_position_rad", "real_repeat_position_rmse_rad"),
                ("present_velocity_rad_s", "real_repeat_velocity_rmse_rad_s"),
                ("present_current_A", "real_repeat_current_rmse_A"),
            ):
                left_value = left.columns[column][mask]
                right_value = np.interp(common, right_t, right.columns[column])
                row[metric] = float(np.sqrt(np.mean((left_value - right_value) ** 2)))
            pair_rows.append(row)
    metric_names = (
        "real_repeat_position_rmse_rad", "real_repeat_velocity_rmse_rad_s",
        "real_repeat_current_rmse_A",
    )
    means = {
        name: float(np.mean([float(row[name]) for row in pair_rows])) if pair_rows else math.nan
        for name in metric_names
    }
    return {"alignment": "run-local host_time linear interpolation onto left-repeat samples",
            "pairs": pair_rows, "mean": means}


def estimate_ad(prior_ap: float, delay_s: float, runs: list[Run]) -> dict[str, Any]:
    values = []
    for run in runs:
        if run.trajectory != "accelerated_oscillation":
            continue
        c = run.columns
        t = c["host_time_ns"] * 1e-9
        event = c["command_event"] == 1
        command_t = c["command_tx_after_ns"][event] * 1e-9
        command_goal = c["goal_position_rad"][event]
        delayed_goal = zoh_command(t - delay_s, command_t, command_goal)
        x = -c["present_velocity_rad_s"]
        y = c["present_current_A"] - prior_ap * (delayed_goal - c["present_position_rad"])
        valid = np.isfinite(x) & np.isfinite(y) & (c["current_saturated"] == 0) & (c["pwm_saturated"] == 0)
        p, sigma, rmse = _robust_linear(np.column_stack((x[valid], np.ones(np.count_nonzero(valid)))), y[valid])
        values.append({"mechanical_configuration": run.mechanical, "repeat": run.repeat,
                       "aD_A_s_per_rad": float(p[0]), "uncertainty": float(sigma[0]), "rmse_A": rmse})
    accepted = [item["aD_A_s_per_rad"] for item in values if item["aD_A_s_per_rad"] >= 0]
    if not accepted:
        raise ValueError("nonnegative aD initial candidate를 얻지 못했습니다.")
    return {"aD_initial_A_s_per_rad": float(np.median(accepted)), "per_run": values}


def _metrics(run: Run, sim: Any, q: np.ndarray, qd: np.ndarray, current: np.ndarray,
             current_saturated: np.ndarray, pwm_saturated: np.ndarray) -> dict[str, Any]:
    sampled_time = sim.time_sec
    peak_real = int(np.argmax(np.abs(qd)))
    peak_sim = int(np.argmax(np.abs(sim.qd_rad_s)))
    current_cap = max(1e-12, float(np.max(np.abs(current))))
    return {"mechanical_configuration": run.mechanical, "trajectory": run.trajectory, "repeat": run.repeat,
            "position_mae_rad": float(np.mean(np.abs(sim.q_rad - q))),
            "position_rmse_rad": float(np.sqrt(np.mean((sim.q_rad - q) ** 2))),
            "velocity_mae_rad_s": float(np.mean(np.abs(sim.qd_rad_s - qd))),
            "current_mae_A": float(np.mean(np.abs(sim.current_A - current))),
            "normalized_current_mae": float(np.mean(np.abs(sim.current_A - current)) / current_cap),
            "sample_count": int(len(q)),
            "current_saturated_sample_count": int(np.count_nonzero(current_saturated)),
            "pwm_saturated_sample_count": int(np.count_nonzero(pwm_saturated)),
            "peak_timing_error_sec": float(sampled_time[peak_sim] - sampled_time[peak_real]),
            "steady_state_error_rad": float(np.mean(sim.q_rad[-max(1, len(q)//10):] - q[-max(1, len(q)//10):]))}


def fit(cfg: CanonicalCampaign, fit_config_path: Path) -> Path:
    if cfg.holdout_configuration is None:
        raise ValueError("holdout_configuration을 데이터 확인 전에 고정해야 합니다.")
    fit_cfg = _load_fit_config(fit_config_path.expanduser().resolve())
    static = estimate_static(cfg, _require_runs(cfg, "static"), fit_cfg["static_bootstrap"])
    delay = estimate_delay(cfg, _require_runs(cfg, "delay")[0])
    dynamic = _require_runs(cfg, "dynamic")
    fit_runs = [run for run in dynamic if run.mechanical != cfg.holdout_configuration]
    if len(fit_runs) != 45:
        raise AssertionError(f"fit run count={len(fit_runs)}, expected=45")
    ad = estimate_ad(float(static["aP_prior_A_per_rad"]), float(delay["delay_median_s"]), fit_runs)
    names = ("aD_A_s_per_rad", "armature_kg_m2", "coulomb_friction_Nm", "viscous_friction_Nm_s_per_rad")
    bounds_cfg = fit_cfg["stage_d"]["parameter_bounds"]
    lower = np.asarray([float(bounds_cfg[name][0]) for name in names])
    upper = np.asarray([float(bounds_cfg[name][1]) for name in names])
    initial_cfg = fit_cfg["stage_d"]["initial_parameters"]
    initial = np.asarray([float(ad["aD_initial_A_s_per_rad"]), float(initial_cfg["armature_kg_m2"]), float(initial_cfg["coulomb_friction_Nm"]), float(initial_cfg["viscous_friction_Nm_s_per_rad"])])
    dt = float(fit_cfg["physics_timestep_sec"])
    stride = int(fit_cfg["stage_d"]["evaluation_stride"])
    scales = np.asarray([float(fit_cfg["loss"]["position_scale_rad"]), float(fit_cfg["loss"]["velocity_scale_rad_s"]), float(fit_cfg["loss"]["current_scale_A"])])
    weights = fit_cfg["loss"]["weights"]
    prepared = [(run, _dynamic_arrays(run, stride)) for run in fit_runs]
    validity = primary_fit_sample_counts(prepared)
    if validity["total_sample_count"] == validity["excluded_from_primary_fit_count"]:
        raise ValueError("Stage D primary fit에 사용할 non-PWM-saturated sample이 없습니다.")

    def parameters(x: np.ndarray) -> dict[str, float]:
        return {"aP_A_per_rad": float(static["aP_prior_A_per_rad"]), "Ktau_eff_Nm_per_A": float(static["Ktau_eff_prior_Nm_per_A"]),
                "delay_s": float(delay["delay_median_s"]), **{name: float(value) for name, value in zip(names, x)}}

    def simulation_residual(p: dict[str, float]) -> np.ndarray:
        parts = []
        for run, arrays in prepared:
            t, cmd_t, cmd, q, qd, current, _current_sat, pwm_sat = arrays
            sim = replay(cfg, run.mechanical, t, cmd_t, cmd, q[0], qd[0], p, dt)
            primary = ~pwm_sat
            parts.extend((math.sqrt(float(weights["position"])) * (sim.q_rad[primary] - q[primary]) / scales[0],
                          math.sqrt(float(weights["velocity"])) * (sim.qd_rad_s[primary] - qd[primary]) / scales[1],
                          math.sqrt(float(weights["current"])) * (sim.current_A[primary] - current[primary]) / scales[2]))
        return np.concatenate(parts)

    def residual(x: np.ndarray) -> np.ndarray:
        return simulation_residual(parameters(x))

    candidates = []
    for index, seed in enumerate(fit_cfg["stage_d"]["independent_seeds"]):
        x0 = initial if index == 0 else np.random.default_rng(int(seed)).uniform(lower, upper)
        result = least_squares(residual, np.clip(x0, lower, upper), bounds=(lower, upper),
                               max_nfev=int(fit_cfg["stage_d"]["max_function_evaluations"]), verbose=1)
        candidates.append(result)
    best = min(candidates, key=lambda item: float(np.mean(item.fun**2)))
    params = parameters(best.x)
    stage_e_candidates = []
    selected_stage = "D"
    if bool(fit_cfg["stage_e"].get("enabled")):
        stage_e_names = tuple(fit_cfg["stage_e"]["optimized_parameters"])
        expected = (
            "aP_A_per_rad", "aD_A_s_per_rad", "Ktau_eff_Nm_per_A", "delay_s",
            "armature_kg_m2", "coulomb_friction_Nm", "viscous_friction_Nm_s_per_rad",
        )
        if stage_e_names != expected:
            raise ValueError(f"stage_e.optimized_parameters 순서는 {expected}여야 합니다.")
        stage_e_bounds = fit_cfg["stage_e"]["uncertainty_bounds"]
        missing_bounds = [name for name in stage_e_names if name not in stage_e_bounds]
        if missing_bounds:
            raise ValueError("Stage E uncertainty bound 누락: " + ", ".join(missing_bounds))
        stage_e_lower = np.asarray([float(stage_e_bounds[name][0]) for name in stage_e_names])
        stage_e_upper = np.asarray([float(stage_e_bounds[name][1]) for name in stage_e_names])
        if np.any(stage_e_lower >= stage_e_upper):
            raise ValueError("Stage E uncertainty bound는 모두 lower < upper여야 합니다.")
        stage_e_initial = np.asarray([float(params[name]) for name in stage_e_names])
        prior_scales = {
            "aP_A_per_rad": max(float(static["aP_uncertainty_A_per_rad"]), np.finfo(float).eps),
            "Ktau_eff_Nm_per_A": max(float(static["Ktau_static_uncertainty_Nm_per_A"]), np.finfo(float).eps),
            "delay_s": max(float(delay["delay_std_s"]), float(delay["sampling_resolution_s"])),
        }

        def stage_e_parameters(x: np.ndarray) -> dict[str, float]:
            return {name: float(value) for name, value in zip(stage_e_names, x)}

        def stage_e_residual(x: np.ndarray) -> np.ndarray:
            p = stage_e_parameters(x)
            regularization = np.asarray([
                (p["aP_A_per_rad"] - float(static["aP_prior_A_per_rad"])) / prior_scales["aP_A_per_rad"],
                (p["Ktau_eff_Nm_per_A"] - float(static["Ktau_eff_prior_Nm_per_A"])) / prior_scales["Ktau_eff_Nm_per_A"],
                (p["delay_s"] - float(delay["delay_median_s"])) / prior_scales["delay_s"],
            ])
            return np.concatenate((simulation_residual(p), regularization))

        for index, seed in enumerate(fit_cfg["stage_d"]["independent_seeds"]):
            x0 = stage_e_initial if index == 0 else np.random.default_rng(int(seed)).uniform(stage_e_lower, stage_e_upper)
            result = least_squares(
                stage_e_residual,
                np.clip(x0, stage_e_lower, stage_e_upper),
                bounds=(stage_e_lower, stage_e_upper),
                max_nfev=int(fit_cfg["stage_d"]["max_function_evaluations"]),
                verbose=1,
            )
            stage_e_candidates.append(result)
        stage_e_best = min(stage_e_candidates, key=lambda item: float(np.mean(item.fun**2)))
        params = stage_e_parameters(stage_e_best.x)
        selected_stage = "E"
    params["derived"] = {"Kp_eq_Nm_per_rad": params["Ktau_eff_Nm_per_A"] * params["aP_A_per_rad"],
                         "Kd_eq_Nm_s_per_rad": params["Ktau_eff_Nm_per_A"] * params["aD_A_s_per_rad"]}
    params["current_limit_A"] = int(cfg.registers["expected_current_limit_raw"]) * 0.00336
    params["goal_current_limit_A"] = abs(int(cfg.registers["goal_current_raw"])) * 0.00336
    params["model"] = "mode5_current_domain_m1"
    params["selected_fit_stage"] = selected_stage
    params["controller_registers"] = cfg.registers
    output = cfg.results_root / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_mode5_m1")
    (output / "plots").mkdir(parents=True, exist_ok=False)
    (output / "params_mode5_m1.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
    (output / "static_calibration.json").write_text(json.dumps(static, indent=2, ensure_ascii=False) + "\n")
    (output / "delay_calibration.json").write_text(json.dumps(delay, indent=2, ensure_ascii=False) + "\n")
    (output / "parameter_uncertainty.json").write_text(json.dumps({
        "static": {k: v for k, v in static.items() if "uncertainty" in k},
        "delay_std_s": delay["delay_std_s"],
        "stage_d_optimizer_seed_solutions": [item.x.tolist() for item in candidates],
        "stage_e_enabled": bool(stage_e_candidates),
        "stage_e_optimizer_seed_solutions": [item.x.tolist() for item in stage_e_candidates],
    }, indent=2) + "\n")
    fit_metrics = []
    for run, arrays in prepared:
        t, cmd_t, cmd, q, qd, current, current_sat, pwm_sat = arrays
        fit_metrics.append(_metrics(run, replay(cfg, run.mechanical, t, cmd_t, cmd, q[0], qd[0], params, dt), q, qd, current, current_sat, pwm_sat))
    (output / "metrics_fit.json").write_text(json.dumps(fit_metrics, indent=2) + "\n")
    (output / "fit_validity_region.json").write_text(json.dumps({
        "primary_stage_d_policy": "exclude PWM-saturated samples; retain current-saturated samples because M1 models current clipping",
        **validity,
    }, indent=2) + "\n")
    (output / "real_to_real_repeatability.json").write_text(
        json.dumps(real_to_real_repeatability(dynamic), indent=2) + "\n"
    )
    fit_path = fit_config_path.resolve()
    (output / "manifest.json").write_text(json.dumps({"config": cfg.config_manifest(),
        "fit_config": {"path": str(fit_path), "sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest()},
        "holdout_configuration": cfg.holdout_configuration, "fit_runs": 45, "held_out_runs": 9,
        "holdout_scope": "dynamic trajectories only; static calibration uses all six configurations",
        "selected_fit_stage": selected_stage, "primary_fit_validity": validity}, indent=2) + "\n")
    (output / "residual_summary.json").write_text(json.dumps({"status": "pending held-out validation"}, indent=2) + "\n")
    _plot_static_diagnostics(static, output / "plots")
    return output


def _plot_static_diagnostics(static: dict[str, Any], plots: Path) -> None:
    """Plot diagnostics only; model extension remains a manual decision."""
    rows = static["diagnostic_rows"]
    x_fields = (
        ("required_torque", "|gravity torque| [Nm]", True),
        ("mass_kg", "load mass [kg]", False),
        ("arm_length_m", "arm length [m]", False),
        ("branch", "approach direction (+/-)", False),
        ("q", "angle [rad]", False),
        ("current", "Present Current [A]", False),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (field, label, absolute) in zip(axes.flat, x_fields):
        x = np.asarray([float(row[field]) for row in rows])
        if absolute:
            x = np.abs(x)
        y = np.asarray([float(row["torque_residual_Nm"]) for row in rows])
        ax.scatter(x, y, s=12, alpha=0.55)
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_xlabel(label)
        ax.set_ylabel("static torque residual [Nm]")
    fig.suptitle("Static M1 residual diagnostics (model extension: manual review)")
    fig.tight_layout()
    fig.savefig(plots / "static_residual_diagnostics.png", dpi=160)
    plt.close(fig)


def _result_fit_config(cfg: CanonicalCampaign, target: Path) -> dict[str, Any]:
    manifest = json.loads((target / "manifest.json").read_text())
    if manifest.get("holdout_configuration") != cfg.holdout_configuration:
        raise ValueError("현재 campaign holdout이 fit 당시 manifest와 다릅니다.")
    if manifest.get("config") != cfg.config_manifest():
        raise ValueError("현재 canonical config/hash가 fit 당시 manifest와 다릅니다.")
    info = manifest["fit_config"]
    path = Path(info["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError("fit config가 식별 이후 변경되었습니다.")
    return _load_fit_config(path)


def validate(cfg: CanonicalCampaign, result_dir: Path) -> Path:
    target = result_dir.expanduser().resolve()
    params = yaml.safe_load((target / "params_mode5_m1.yaml").read_text())
    fit_cfg = _result_fit_config(cfg, target)
    dt, stride = float(fit_cfg["physics_timestep_sec"]), int(fit_cfg["stage_d"]["evaluation_stride"])
    holdout = [run for run in _require_runs(cfg, "dynamic") if run.mechanical == cfg.holdout_configuration]
    if len(holdout) != 9:
        raise AssertionError(f"held-out run count={len(holdout)}, expected=9")
    metrics = []
    for run in holdout:
        t, cmd_t, cmd, q, qd, current, current_sat, pwm_sat = _dynamic_arrays(run, stride)
        metrics.append(_metrics(run, replay(cfg, run.mechanical, t, cmd_t, cmd, q[0], qd[0], params, dt), q, qd, current, current_sat, pwm_sat))
    path = target / "metrics_validation.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    return path


def report(cfg: CanonicalCampaign, result_dir: Path) -> Path:
    target = result_dir.expanduser().resolve()
    fit_metrics = json.loads((target / "metrics_fit.json").read_text())
    validation_metrics = json.loads((target / "metrics_validation.json").read_text())
    params = yaml.safe_load((target / "params_mode5_m1.yaml").read_text())
    static = json.loads((target / "static_calibration.json").read_text())
    plots = target / "plots"
    plots.mkdir(exist_ok=True)
    labels = [f"{m['trajectory']} r{m['repeat']}" for m in validation_metrics]
    values = [m["position_rmse_rad"] for m in validation_metrics]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(values)), values)
    ax.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right")
    ax.set_ylabel("position RMSE [rad]")
    ax.set_title(f"Dynamic trajectory holdout: {cfg.holdout_configuration}")
    fig.tight_layout()
    fig.savefig(plots / "heldout_position_rmse.png", dpi=160)
    plt.close(fig)
    fit_cfg = _result_fit_config(cfg, target)
    dt = float(fit_cfg["physics_timestep_sec"])
    stride = int(fit_cfg["stage_d"]["evaluation_stride"])
    residual_rows: dict[str, list[np.ndarray]] = {
        name: [] for name in ("position", "velocity", "current", "gravity", "arm_length", "direction", "pwm", "temperature", "residual")
    }
    for run in [item for item in _require_runs(cfg, "dynamic") if item.mechanical == cfg.holdout_configuration]:
        t, cmd_t, cmd, q, qd, current, _current_sat, _pwm_sat = _dynamic_arrays(run, stride)
        sim = replay(cfg, run.mechanical, t, cmd_t, cmd, q[0], qd[0], params, dt)
        index = np.arange(0, len(run.columns["present_pwm_fraction"]), stride)[:len(q)]
        moment = float(cfg.geometry["gravity_m_s2"]) * (
            float(cfg.geometry["arm_mass_kg"]) * float(cfg.geometry["arm_com_radius_m"])
            + cfg.load_mass_kg(run.mechanical) * cfg.arm_length_m(run.mechanical)
        )
        gravity = np.abs(moment * np.sin(q + float(cfg.geometry["gravity_zero_angle_rad"])))
        residual_rows["position"].append(q)
        residual_rows["velocity"].append(qd)
        residual_rows["current"].append(current)
        residual_rows["gravity"].append(gravity)
        residual_rows["arm_length"].append(np.full(len(q), cfg.arm_length_m(run.mechanical)))
        residual_rows["direction"].append(np.sign(qd))
        residual_rows["pwm"].append(run.columns["present_pwm_fraction"][index])
        residual_rows["temperature"].append(run.columns["temperature_C"][index])
        residual_rows["residual"].append(sim.q_rad - q)
    merged = {name: np.concatenate(parts) for name, parts in residual_rows.items()}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    x_names = ("position", "velocity", "current", "gravity", "arm_length", "direction", "pwm", "temperature")
    correlations = {}
    for ax, name in zip(axes.flat, x_names):
        ax.scatter(merged[name], merged["residual"], s=2, alpha=0.25)
        ax.set_xlabel(name)
        ax.set_ylabel("q_sim - q_real [rad]")
        correlations[name] = float(np.corrcoef(merged[name], merged["residual"])[0, 1]) if np.std(merged[name]) > 0 else 0.0
    fig.tight_layout()
    fig.savefig(plots / "heldout_residual_diagnostics.png", dpi=160)
    plt.close(fig)
    model_variation: dict[str, float] = {}
    for trajectory in sorted(set(item["trajectory"] for item in validation_metrics)):
        selected = [item["position_rmse_rad"] for item in validation_metrics if item["trajectory"] == trajectory]
        model_variation[trajectory] = float(np.std(selected, ddof=1)) if len(selected) > 1 else 0.0
    real_repeatability = json.loads((target / "real_to_real_repeatability.json").read_text())
    residual = {"fit_position_rmse_mean_rad": float(np.mean([m["position_rmse_rad"] for m in fit_metrics])),
                "validation_position_rmse_mean_rad": float(np.mean(values)),
                "holdout_configuration": cfg.holdout_configuration,
                "heldout_residual_correlations": correlations,
                "model_error_repeat_variation_position_rmse_std_rad": model_variation,
                "real_to_real_repeatability_mean": real_repeatability["mean"],
                "model_extension_decision": "manual residual review required"}
    (target / "residual_summary.json").write_text(json.dumps(residual, indent=2) + "\n")
    text = ["# Mode 5 current-domain M1 report", "", f"- Dynamic trajectory holdout configuration: {cfg.holdout_configuration}",
            f"- Fit runs: {len(fit_metrics)}", f"- Validation runs: {len(validation_metrics)}",
            f"- Static accepted/rejected plateaus: {static['accepted_plateau_count']}/{static['rejected_plateau_count']}",
            f"- Static reject reasons: {static['reject_reason_counts']}",
            f"- Fit mean position RMSE: {residual['fit_position_rmse_mean_rad']:.6g} rad",
            f"- Validation mean position RMSE: {residual['validation_position_rmse_mean_rad']:.6g} rad", "",
            f"Static diagnostic: {static['diagnostic_note']}",
            "Whole-Jandi parameter update is not automatic. Review trajectory/configuration metrics, repeatability, saturation, thermal drift, and residual plots first.", "",
            "```yaml", yaml.safe_dump(params, sort_keys=False).rstrip(), "```"]
    path = target / "report.md"
    path.write_text("\n".join(text) + "\n")
    return path
