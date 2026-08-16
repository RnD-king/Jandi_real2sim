from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.signal import savgol_filter

from .config import CONDITIONS, TRAJECTORIES, Mode5Campaign


@dataclass(frozen=True)
class Run:
    condition: str
    trajectory: str
    repeat: int
    path: Path
    columns: dict[str, np.ndarray]
    events: dict[str, np.ndarray]


def _source_manifest(cfg: Mode5Campaign) -> dict[str, dict[str, str]]:
    return {
        role: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for role, path in sorted(cfg.source_files.items())
    }


def _float(value: str) -> float:
    return float(value) if value != "" else math.nan


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"빈 telemetry: {path}")
    numeric = {
        name: np.asarray([_float(row[name]) for row in rows], dtype=float)
        for name in (
            "host_time_ns", "tx_start_ns", "cycle_index", "time_s", "overrun_ns",
            "q_cmd_rad", "q_present_rad", "dq_present_rad_s", "current_A_joint",
            "pwm_percent", "input_voltage_V", "temperature_C",
        )
    }
    numeric["phase"] = np.asarray([row["phase"] for row in rows], dtype=object)
    numeric["kind"] = np.asarray([row["acquisition_kind"] for row in rows], dtype=object)
    return numeric


def _load_events(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in ("sequence", "goal_rad", "tx_start_ns", "tx_end_ns")
    }


def load_runs(cfg: Mode5Campaign) -> list[Run]:
    root = cfg.output_root / cfg.campaign_id
    result: list[Run] = []
    missing: list[Path] = []
    for condition in CONDITIONS:
        for trajectory in TRAJECTORIES:
            for repeat in cfg.repeats:
                path = root / condition / trajectory / f"repeat_{repeat}"
                metadata_path = path / "metadata.json"
                telemetry_path = path / "telemetry.csv"
                if not metadata_path.exists() or not telemetry_path.exists():
                    missing.append(path)
                    continue
                metadata = json.loads(metadata_path.read_text())
                if not metadata.get("valid_flag"):
                    raise ValueError(f"invalid run: {path}: {metadata.get('invalid_reason')}")
                event_path = path / "command_events.csv"
                if not event_path.exists():
                    missing.append(path)
                    continue
                result.append(
                    Run(
                        condition,
                        trajectory,
                        repeat,
                        path,
                        _load_csv(telemetry_path),
                        _load_events(event_path),
                    )
                )
    if missing:
        raise FileNotFoundError(
            "18개 고정 run 중 누락:\n" + "\n".join(str(path) for path in missing)
        )
    return result


def _state(run: Run) -> dict[str, np.ndarray]:
    mask = run.columns["kind"] == "state"
    return {name: values[mask] for name, values in run.columns.items()}


def estimate_delay(runs: list[Run]) -> dict[str, float | list[float]]:
    delays: list[float] = []
    for run in runs:
        if run.trajectory != "step" or run.repeat == 3:
            continue
        data = _state(run)
        current = data["current_A_joint"]
        time_ns = data["host_time_ns"]
        event_changes = np.flatnonzero(np.abs(np.diff(run.events["goal_rad"])) > 1e-7) + 1
        for event_index in event_changes:
            tx_ns = run.events["tx_end_ns"][event_index]
            before = current[(time_ns < tx_ns) & (time_ns >= tx_ns - 0.25e9)]
            if len(before) < 5:
                continue
            baseline = float(np.median(before))
            noise = 1.4826 * float(np.median(np.abs(before - baseline)))
            threshold = max(0.015, 4.0 * noise)
            after_indices = np.flatnonzero((time_ns >= tx_ns) & (time_ns <= tx_ns + 0.20e9))
            response = np.flatnonzero(
                np.abs(current[after_indices] - baseline) >= threshold
            )
            if response.size:
                sample_index = after_indices[int(response[0])]
                delays.append((time_ns[sample_index] - tx_ns) * 1e-9)
    if not delays:
        raise ValueError("step current onset에서 delay를 계산하지 못했습니다.")
    array = np.asarray(delays)
    return {
        "samples_s": [float(value) for value in array],
        "median_s": float(np.median(array)),
        "mean_s": float(np.mean(array)),
        "std_s": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min_s": float(np.min(array)),
        "max_s": float(np.max(array)),
        "count": len(array),
    }


def _plateaus(run: Run) -> list[dict[str, float]]:
    data = _state(run)
    phases = data["phase"]
    result = []
    for phase in dict.fromkeys(phases.tolist()):
        indices = np.flatnonzero(phases == phase)
        if len(indices) < 10:
            continue
        tail = indices[round(len(indices) * 0.65):]
        result.append({
            "q_cmd": float(np.median(data["q_cmd_rad"][tail])),
            "q": float(np.median(data["q_present_rad"][tail])),
            "dq": float(np.median(data["dq_present_rad_s"][tail])),
            "current": float(np.median(data["current_A_joint"][tail])),
        })
    return result


def estimate_static(cfg: Mode5Campaign, runs: list[Run]) -> dict[str, object]:
    fit_steps = [run for run in runs if run.trajectory == "step" and run.repeat in (1, 2)]
    rows: list[tuple[float, float, float]] = []
    loaded_rows: list[tuple[float, float]] = []
    for run in fit_steps:
        bench = cfg.benches[run.condition]
        assert bench.gravity_zero_offset_rad is not None
        assert bench.arm_mass_kg is not None and bench.arm_com_radius_m is not None
        added_mass = bench.added_load_mass_kg or 0.0
        added_radius = bench.added_load_radius_m or 0.0
        gravity_gain = 9.80665 * (
            bench.arm_mass_kg * bench.arm_com_radius_m + added_mass * added_radius
        )
        for point in _plateaus(run):
            error = point["q_cmd"] - point["q"]
            required_torque = gravity_gain * math.sin(
                point["q"] + bench.gravity_zero_offset_rad
            )
            rows.append((error, point["current"], required_torque))
            if run.condition == "loaded":
                loaded_rows.append((point["current"], required_torque))
    array = np.asarray(rows)
    if len(array) < 8:
        raise ValueError("static regression 표본이 부족합니다.")
    current_fit = np.column_stack((array[:, 0], np.ones(len(array))))
    a_p, current_bias = np.linalg.lstsq(current_fit, array[:, 1], rcond=None)[0]
    loaded_array = np.asarray(loaded_rows)
    if len(loaded_array) < 4:
        raise ValueError("Ktau 계산을 위한 loaded step plateau가 부족합니다.")
    torque_fit = np.column_stack((loaded_array[:, 0], np.ones(len(loaded_array))))
    k_tau, torque_bias = np.linalg.lstsq(torque_fit, loaded_array[:, 1], rcond=None)[0]
    return {
        "aP_A_per_rad": float(a_p),
        "current_bias_A": float(current_bias),
        "Ktau_Nm_per_A": float(k_tau),
        "torque_bias_Nm": float(torque_bias),
        "plateau_count": len(array),
    }


def estimate_ad(runs: list[Run], delay_s: float) -> dict[str, object]:
    def matrix(selected: list[Run]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for run in selected:
            data = _state(run)
            shift = max(0, round(delay_s * 100))
            q_cmd = np.roll(data["q_cmd_rad"], shift)
            q_cmd[:shift] = q_cmd[shift] if shift else q_cmd[:shift]
            e = q_cmd - data["q_present_rad"]
            valid = np.isfinite(e) & np.isfinite(data["dq_present_rad_s"]) & np.isfinite(data["current_A_joint"])
            xs.append(np.column_stack((e[valid], -data["dq_present_rad_s"][valid], np.ones(np.count_nonzero(valid)))))
            ys.append(data["current_A_joint"][valid])
        return np.vstack(xs), np.concatenate(ys)

    fit_runs = [run for run in runs if run.trajectory == "sine" and run.repeat in (1, 2)]
    val_runs = [run for run in runs if run.trajectory == "sine" and run.repeat == 3]
    x_fit, y_fit = matrix(fit_runs)
    p_only = np.linalg.lstsq(x_fit[:, (0, 2)], y_fit, rcond=None)[0]
    pd = np.linalg.lstsq(x_fit, y_fit, rcond=None)[0]
    x_val, y_val = matrix(val_runs)
    p_rmse = float(np.sqrt(np.mean((y_val - x_val[:, (0, 2)] @ p_only) ** 2)))
    pd_rmse = float(np.sqrt(np.mean((y_val - x_val @ pd) ** 2)))
    improvement = (p_rmse - pd_rmse) / p_rmse if p_rmse else 0.0
    accepted = bool(pd[1] >= 0.0 and improvement >= 0.10)
    return {
        "candidate_aP_A_per_rad": float(pd[0]),
        "candidate_aD_A_s_per_rad": float(pd[1]),
        "bias_A": float(pd[2]),
        "validation_P_only_rmse_A": p_rmse,
        "validation_PD_rmse_A": pd_rmse,
        "relative_improvement": improvement,
        "accepted_by_initial_rule": accepted,
        "selected_aD_A_s_per_rad": float(pd[1]) if accepted else 0.0,
    }


def estimate_mechanics(cfg: Mode5Campaign, runs: list[Run], k_tau: float) -> dict[str, float]:
    matrices, targets = [], []
    for run in runs:
        if run.trajectory not in ("triangle", "sine") or run.repeat == 3:
            continue
        data = _state(run)
        time_s = data["time_s"]
        uniform_time = np.arange(time_s[0], time_s[-1], 1.0 / cfg.timing.command_rate_hz)
        q = np.interp(uniform_time, time_s, data["q_present_rad"])
        current = np.interp(uniform_time, time_s, data["current_A_joint"])
        if len(q) < 21:
            continue
        window = min(31, len(q) - (1 - len(q) % 2))
        if window < 7:
            continue
        dt = 1.0 / cfg.timing.command_rate_hz
        dq = savgol_filter(q, window, 3, deriv=1, delta=dt)
        ddq = savgol_filter(q, window, 3, deriv=2, delta=dt)
        bench = cfg.benches[run.condition]
        assert bench.gravity_zero_offset_rad is not None
        assert bench.arm_mass_kg is not None and bench.arm_com_radius_m is not None
        gravity_gain = 9.80665 * (
            bench.arm_mass_kg * bench.arm_com_radius_m
            + (bench.added_load_mass_kg or 0.0) * (bench.added_load_radius_m or 0.0)
        )
        known_pivot_inertia = bench.equivalent_pivot_inertia_kg_m2
        assert known_pivot_inertia is not None
        y = (
            k_tau * current
            - gravity_gain * np.sin(q + bench.gravity_zero_offset_rad)
            - known_pivot_inertia * ddq
        )
        moving = np.abs(dq) >= 0.015
        matrices.append(np.column_stack((ddq[moving], dq[moving], np.sign(dq[moving]))))
        targets.append(y[moving])
    x = np.vstack(matrices)
    y = np.concatenate(targets)
    params = np.linalg.lstsq(x, y, rcond=None)[0]
    prediction = x @ params
    return {
        "J_eff_kg_m2": float(params[0]),
        "viscous_Nm_s_per_rad": float(params[1]),
        "coulomb_Nm": float(params[2]),
        "fit_torque_rmse_Nm": float(np.sqrt(np.mean((y - prediction) ** 2))),
        "sample_count": int(len(y)),
    }


def fit(cfg: Mode5Campaign) -> Path:
    runs = load_runs(cfg)
    delay = estimate_delay(runs)
    static = estimate_static(cfg, runs)
    ad = estimate_ad(runs, float(delay["median_s"]))
    mechanics = estimate_mechanics(cfg, runs, float(static["Ktau_Nm_per_A"]))
    output = cfg.project_root / "results" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_mode5_minimal"
    )
    output.mkdir(parents=True, exist_ok=False)
    result = {"delay": delay, "static": static, "aD_gate": ad, "mechanics": mechanics}
    (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    parameters = {
        "model": "mode5_current_domain_equivalent_v1",
        "delay_s": delay["median_s"],
        "aP_A_per_rad": static["aP_A_per_rad"],
        "aD_A_s_per_rad": ad["selected_aD_A_s_per_rad"],
        "Ktau_Nm_per_A": static["Ktau_Nm_per_A"],
        "armature_kg_m2": mechanics["J_eff_kg_m2"],
        "frictionloss_Nm": mechanics["coulomb_Nm"],
        "damping_Nm_s_per_rad": mechanics["viscous_Nm_s_per_rad"],
        "current_cap_A": abs(int(cfg.registers.goal_current_raw)) * 0.00336,
    }
    (output / "params_mode5.yaml").write_text(yaml.safe_dump(parameters, sort_keys=False))
    (output / "config_manifest.json").write_text(
        json.dumps(
            {
                "sources": _source_manifest(cfg),
                "benches": {
                    name: bench.resolved_metadata()
                    for name, bench in cfg.benches.items()
                },
                "repeats": list(cfg.repeats),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    report = [
        "# Mode 5 minimal identification report",
        "",
        f"- delay median: {float(delay['median_s']) * 1000:.3f} ms",
        f"- aP: {float(static['aP_A_per_rad']):.6f} A/rad",
        f"- Ktau: {float(static['Ktau_Nm_per_A']):.6f} Nm/A",
        f"- aD candidate accepted: {ad['accepted_by_initial_rule']}",
        f"- selected aD: {float(ad['selected_aD_A_s_per_rad']):.6f} A*s/rad",
        f"- J_eff: {float(mechanics['J_eff_kg_m2']):.8f} kg*m^2",
        f"- Coulomb: {float(mechanics['coulomb_Nm']):.6f} Nm",
        f"- viscous: {float(mechanics['viscous_Nm_s_per_rad']):.6f} Nm*s/rad",
        "",
        "Repeat 1·2만 계산에 사용했고 repeat 3은 aD 채택 판정에만 사용했습니다.",
        "경계값·부호·잔차를 검토하기 전에는 이 값을 물리적 참값으로 확정하지 마십시오.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    return output


def compare(cfg: Mode5Campaign, params_path: Path, output: Path | None = None) -> Path:
    from .mujoco_model import replay

    runs = load_runs(cfg)
    params = yaml.safe_load(params_path.expanduser().read_text())
    target = output or cfg.project_root / "results" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_mode5_data_overview"
    )
    target.mkdir(parents=True, exist_ok=False)
    metrics: dict[str, dict[str, float | str | int]] = {}
    for run in runs:
        result = replay(cfg, run, params)
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        axes[0].plot(result.time_s, result.q_cmd_delayed, "k--", label="q_cmd delayed")
        axes[0].plot(result.time_s, result.q_real, label="q_real")
        axes[0].plot(result.time_s, result.q_sim, label="q_sim")
        axes[0].set_ylabel("angle [rad]")
        axes[0].legend()
        axes[1].plot(result.time_s, result.dq_real, label="dq_real")
        axes[1].plot(result.time_s, result.dq_sim, label="dq_sim")
        axes[1].set_ylabel("velocity [rad/s]")
        axes[1].legend()
        axes[2].plot(result.time_s, result.current_real, label="I_real")
        axes[2].plot(result.time_s, result.current_model, label="I_model")
        axes[2].set_ylabel("joint current [A]")
        axes[2].set_xlabel("time [s]")
        axes[2].legend()
        fig.suptitle(f"{run.condition} / {run.trajectory} / repeat {run.repeat}")
        fig.tight_layout()
        fig.savefig(target / f"{run.condition}_{run.trajectory}_repeat{run.repeat}.png", dpi=140)
        plt.close(fig)
        key = f"{run.condition}/{run.trajectory}/repeat_{run.repeat}"
        metrics[key] = {
            "condition": run.condition,
            "trajectory": run.trajectory,
            "repeat": run.repeat,
            "split_role": "validation" if run.repeat == 3 else "fit",
            "q_mae_rad": float(np.mean(np.abs(result.q_real - result.q_sim))),
            "q_rmse_rad": float(np.sqrt(np.mean((result.q_real - result.q_sim) ** 2))),
            "dq_mae_rad_s": float(np.mean(np.abs(result.dq_real - result.dq_sim))),
            "current_mae_A": float(np.mean(np.abs(result.current_real - result.current_model))),
        }
    (target / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "campaign": str(cfg.source),
                "config_sources": _source_manifest(cfg),
                "parameters": str(params_path.expanduser().resolve()),
                "run_count": len(runs),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return target
