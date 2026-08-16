from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import minimize

from jandi_real2sim.config import MUJOCO_DOF_ORDER
from jandi_real2sim.identification.dataset import (
    RunData,
    load_run,
    split_fit_validation,
)
from jandi_real2sim.identification.fit_m0 import _huber
from jandi_real2sim.identification.replay import (
    FixedBaseReplay,
    ReplayConfig,
    ReplayResult,
)


@dataclass(frozen=True)
class GainCondition:
    name: str
    register_p: int
    initial_kp: float
    initial_kd: float


@dataclass(frozen=True)
class DualGainFitConfig:
    source: Path
    replay: ReplayConfig
    target_joints: tuple[str, str]
    pose_id: str
    conditions: tuple[GainCondition, GainCondition]
    kp_bounds: tuple[float, float]
    kd_bounds: tuple[float, float]
    delay_values_sec: np.ndarray
    transient_window_sec: float
    plateau_tail_sec: float
    huber_delta_normalized: float
    velocity_loss_weight: float
    optimizer_maxiter: int
    optimizer_max_evaluations: int


@dataclass(frozen=True)
class PdFit:
    kp: float
    kd: float
    loss: float
    success: bool
    evaluations: int


@dataclass(frozen=True)
class SharedDelayCandidate:
    delay_sec: float
    by_condition: dict[str, PdFit]
    joint_loss: float


def load_dual_gain_fit_config(path: str | Path) -> DualGainFitConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text())
    kp_baseline = np.asarray(raw["kp_baseline"], dtype=np.float64)
    kd_baseline = np.asarray(raw["kd_baseline"], dtype=np.float64)
    if kp_baseline.shape != (12,) or kd_baseline.shape != (12,):
        raise ValueError("kp_baseline/kd_baseline은 12개 값이어야 합니다.")
    target_joints = tuple(str(value) for value in raw["target_joints"])
    if len(target_joints) != 2 or any(joint not in MUJOCO_DOF_ORDER for joint in target_joints):
        raise ValueError("target_joints에는 유효한 두 관절이 필요합니다.")
    condition_raw = raw["gain_conditions"]
    if len(condition_raw) != 2:
        raise ValueError("gain_conditions는 정확히 두 조건이어야 합니다.")
    conditions = tuple(
        GainCondition(
            name=str(name),
            register_p=int(values["register_p"]),
            initial_kp=float(values["initial_kp"]),
            initial_kd=float(values["initial_kd"]),
        )
        for name, values in condition_raw.items()
    )
    delay_step = float(raw["delay_step_sec"])
    delay_values = np.arange(
        float(raw["delay_min_sec"]),
        float(raw["delay_max_sec"]) + 0.5 * delay_step,
        delay_step,
        dtype=np.float64,
    )
    physics_dt = float(raw["physics_dt_sec"])
    if np.any(~np.isclose(delay_values / physics_dt, np.round(delay_values / physics_dt))):
        raise ValueError("delay 후보는 physics_dt의 정수배여야 합니다.")
    transient_window = float(raw["transient_window_sec"])
    plateau_tail = float(raw["plateau_tail_sec"])
    if transient_window <= 0.0 or plateau_tail <= 0.0:
        raise ValueError("transient/plateau 시간은 양수여야 합니다.")
    return DualGainFitConfig(
        source=source,
        replay=ReplayConfig(
            model_xml=Path(raw["model_xml"]),
            physics_dt_sec=physics_dt,
            torque_limit_nm=float(raw["torque_limit_nm"]),
            kp_baseline=kp_baseline,
            kd_baseline=kd_baseline,
        ),
        target_joints=(target_joints[0], target_joints[1]),
        pose_id=str(raw["pose_id"]),
        conditions=(conditions[0], conditions[1]),
        kp_bounds=tuple(map(float, raw["kp_bounds"])),
        kd_bounds=tuple(map(float, raw["kd_bounds"])),
        delay_values_sec=delay_values,
        transient_window_sec=transient_window,
        plateau_tail_sec=plateau_tail,
        huber_delta_normalized=float(raw["huber_delta_normalized"]),
        velocity_loss_weight=float(raw["velocity_loss_weight"]),
        optimizer_maxiter=int(raw["optimizer_maxiter"]),
        optimizer_max_evaluations=int(raw["optimizer_max_evaluations"]),
    )


def load_collection_campaign_runs(
    campaign_root: str | Path,
    config: DualGainFitConfig,
    expected_register_p: int,
) -> tuple[RunData, ...]:
    """collection campaign status의 정확한 이름만 사용해 6개 step run을 읽는다."""
    root = Path(campaign_root).expanduser().resolve()
    status = json.loads((root / "campaign_status.json").read_text())
    completed = status["completed"]
    runs: list[RunData] = []
    for repeat in (1, 2, 3):
        role = "fit" if repeat in (1, 2) else "validation"
        for joint in config.target_joints:
            key = f"multi_amplitude_step/{joint}/{repeat}/{role}"
            if key not in completed:
                raise ValueError(f"campaign status에 run이 없습니다: {key}")
            run = load_run(root / "runs" / completed[key])
            if run.metadata.get("pose_id") != config.pose_id:
                raise ValueError(f"{run.run_dir}: pose_id 불일치")
            settings = run.metadata.get("actuator_settings", {})
            actual = {
                int(values["position_p_gain"])
                for values in settings.values()
            }
            if actual != {expected_register_p}:
                raise ValueError(
                    f"{run.run_dir}: Position P={actual}, expected={expected_register_p}"
                )
            runs.append(run)
    # 기존 검증 함수가 요구하는 joint/repeat 고유성과 split 계약도 다시 확인한다.
    split_fit_validation(tuple(runs))
    return tuple(runs)


def _edge_segments(run: RunData, config: DualGainFitConfig) -> tuple[tuple[int, int, int, int], ...]:
    command = run.q_cmd_rad[:, run.target_index]
    edges = np.flatnonzero(np.abs(np.diff(command)) > 1e-12) + 1
    transient_count = max(1, round(config.transient_window_sec * run.command_rate_hz))
    tail_count = max(1, round(config.plateau_tail_sec * run.command_rate_hz))
    segments: list[tuple[int, int, int, int]] = []
    for number, edge in enumerate(edges):
        next_edge = int(edges[number + 1]) if number + 1 < len(edges) else len(command)
        end = min(next_edge, int(edge) + transient_count)
        tail_start = max(int(edge), next_edge - tail_count)
        if end > edge and next_edge > tail_start:
            segments.append((int(edge), end, tail_start, next_edge))
    if not segments:
        raise ValueError(f"step edge가 없습니다: {run.run_dir}")
    return tuple(segments)


def dynamic_replay_loss(
    run: RunData,
    result: ReplayResult,
    config: DualGainFitConfig,
) -> float:
    """각 응답을 자기 plateau 기준으로 비교해 static hysteresis를 PD loss에서 분리한다."""
    index = run.target_index
    command = run.q_cmd_rad[:, index]
    q_losses: list[np.ndarray] = []
    dq_losses: list[np.ndarray] = []
    observed_velocity = np.abs(run.dq_real_rad_s[run.state_mask, index])
    dq_scale = max(float(np.nanpercentile(observed_velocity, 95)), 0.05)
    for edge, end, tail_start, tail_end in _edge_segments(run, config):
        transient_mask = run.state_mask[edge:end]
        real_tail_mask = run.state_mask[tail_start:tail_end]
        if not transient_mask.any() or not real_tail_mask.any():
            continue
        real_plateau = float(np.nanmedian(run.q_real_rad[tail_start:tail_end, index][real_tail_mask]))
        sim_plateau = float(np.median(result.q_rad[tail_start:tail_end, index]))
        q_scale = max(abs(float(command[edge] - command[edge - 1])), 1e-3)
        real_q = run.q_real_rad[edge:end, index][transient_mask] - real_plateau
        sim_q = result.q_rad[edge:end, index][transient_mask] - sim_plateau
        q_losses.append(_huber((sim_q - real_q) / q_scale, config.huber_delta_normalized))
        if config.velocity_loss_weight > 0.0:
            real_dq = run.dq_real_rad_s[edge:end, index][transient_mask]
            sim_dq = result.dq_rad_s[edge:end, index][transient_mask]
            dq_losses.append(_huber((sim_dq - real_dq) / dq_scale, config.huber_delta_normalized))
    if not q_losses:
        raise ValueError(f"유효한 transient 표본이 없습니다: {run.run_dir}")
    loss = float(np.mean(np.concatenate(q_losses)))
    if dq_losses:
        loss += config.velocity_loss_weight * float(np.mean(np.concatenate(dq_losses)))
    return loss


def _condition_loss(
    simulator: FixedBaseReplay,
    runs: tuple[RunData, ...],
    config: DualGainFitConfig,
    delay_sec: float,
    kp: float,
    kd: float,
) -> float:
    return float(
        np.mean(
            [
                dynamic_replay_loss(
                    run,
                    simulator.run(run, delay_sec=delay_sec, target_kp=kp, target_kd=kd),
                    config,
                )
                for run in runs
            ]
        )
    )


def fit_shared_delay_candidates(
    simulator: FixedBaseReplay,
    fit_runs_by_condition: dict[str, tuple[RunData, ...]],
    config: DualGainFitConfig,
) -> tuple[SharedDelayCandidate, ...]:
    candidates: list[SharedDelayCandidate] = []
    for delay in config.delay_values_sec:
        by_condition: dict[str, PdFit] = {}
        for condition in config.conditions:
            runs = fit_runs_by_condition[condition.name]

            def objective(parameters: np.ndarray) -> float:
                return _condition_loss(
                    simulator,
                    runs,
                    config,
                    float(delay),
                    float(parameters[0]),
                    float(parameters[1]),
                )

            result = minimize(
                objective,
                x0=np.asarray([condition.initial_kp, condition.initial_kd]),
                method="Powell",
                bounds=(config.kp_bounds, config.kd_bounds),
                options={
                    "maxiter": config.optimizer_maxiter,
                    "maxfev": config.optimizer_max_evaluations,
                    "xtol": 1e-3,
                    "ftol": 1e-5,
                },
            )
            by_condition[condition.name] = PdFit(
                kp=float(result.x[0]),
                kd=float(result.x[1]),
                loss=float(result.fun),
                success=bool(result.success),
                evaluations=int(result.nfev),
            )
        joint_loss = float(np.mean([item.loss for item in by_condition.values()]))
        candidate = SharedDelayCandidate(float(delay), by_condition, joint_loss)
        candidates.append(candidate)
        detail = "  ".join(
            f"{name}:Kp={item.kp:.4f},Kd={item.kd:.4f},loss={item.loss:.8f},n={item.evaluations}"
            for name, item in by_condition.items()
        )
        print(f"delay={delay * 1000.0:4.1f} ms  joint_loss={joint_loss:.8f}  {detail}")
    return tuple(candidates)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_dict(candidate: SharedDelayCandidate) -> dict[str, Any]:
    return {
        "delay_sec": candidate.delay_sec,
        "delay_ms": candidate.delay_sec * 1000.0,
        "joint_loss": candidate.joint_loss,
        "conditions": {
            name: {
                "kp_eff": fit.kp,
                "kd_eff": fit.kd,
                "dynamic_loss": fit.loss,
                "success": fit.success,
                "evaluations": fit.evaluations,
            }
            for name, fit in candidate.by_condition.items()
        },
    }


def _save_plot(
    path: Path,
    run: RunData,
    result: ReplayResult,
    condition_name: str,
    best: SharedDelayCandidate,
) -> None:
    index = run.target_index
    time = np.arange(len(run.q_cmd_rad)) / run.command_rate_hz
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(time, run.q_cmd_rad[:, index], "k--", lw=1.0, label="q_cmd")
    axis.plot(time[run.state_mask], run.q_real_rad[run.state_mask, index], lw=1.2, label="real")
    axis.plot(time, result.q_rad[:, index], color="tab:red", lw=1.0, label="dual M0")
    axis.set(
        xlabel="time [s]",
        ylabel="joint angle [rad]",
        title=f"{condition_name} shared delay={best.delay_sec * 1000:.1f} ms",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def identify_dual_gain_m0(
    campaigns: dict[str, Path],
    config: DualGainFitConfig,
    output_root: str | Path,
) -> Path:
    all_runs: dict[str, tuple[RunData, ...]] = {}
    fit_runs: dict[str, tuple[RunData, ...]] = {}
    validation_runs: dict[str, tuple[RunData, ...]] = {}
    for condition in config.conditions:
        runs = load_collection_campaign_runs(
            campaigns[condition.name], config, condition.register_p
        )
        fit, validation = split_fit_validation(runs)
        all_runs[condition.name] = runs
        fit_runs[condition.name] = fit
        validation_runs[condition.name] = validation

    simulator = FixedBaseReplay(config.replay)
    candidates = fit_shared_delay_candidates(simulator, fit_runs, config)
    best = min(candidates, key=lambda item: item.joint_loss)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(output_root).expanduser().resolve() / f"{timestamp}_m0_dual_gain"
    output.mkdir(parents=True, exist_ok=False)

    metrics: dict[str, Any] = {}
    fitted_results: dict[tuple[str, Path], ReplayResult] = {}
    for condition in config.conditions:
        fit = best.by_condition[condition.name]
        condition_metrics: list[dict[str, Any]] = []
        for run in all_runs[condition.name]:
            result = simulator.run(
                run, delay_sec=best.delay_sec, target_kp=fit.kp, target_kd=fit.kd
            )
            fitted_results[(condition.name, run.run_dir)] = result
            index = run.target_index
            mask = run.state_mask
            error = result.q_rad[mask, index] - run.q_real_rad[mask, index]
            condition_metrics.append(
                {
                    "run_dir": str(run.run_dir),
                    "joint": run.target_joint,
                    "repeat_index": run.repeat_index,
                    "split_role": run.split_role,
                    "dynamic_loss": dynamic_replay_loss(run, result, config),
                    "full_mae_rad": float(np.mean(np.abs(error))),
                    "full_rmse_rad": float(np.sqrt(np.mean(error**2))),
                }
            )
        metrics[condition.name] = condition_metrics
        for role in ("fit", "validation"):
            selected = [item for item in condition_metrics if item["split_role"] == role]
            metrics[f"{condition.name}_{role}_mean"] = {
                key: float(np.mean([item[key] for item in selected]))
                for key in ("dynamic_loss", "full_mae_rad", "full_rmse_rad")
            }

    manifest = {
        "schema_version": 1,
        "method": "shared command delay + P350/P850 separate effective PD",
        "fit_config": str(config.source),
        "fit_config_sha256": _sha256(config.source),
        "model_xml": str(config.replay.model_xml.expanduser().resolve()),
        "model_xml_sha256": _sha256(config.replay.model_xml.expanduser().resolve()),
        "campaigns": {name: str(path.expanduser().resolve()) for name, path in campaigns.items()},
        "runs": {
            name: [
                {
                    "run_dir": str(run.run_dir),
                    "metadata_sha256": _sha256(run.run_dir / "metadata.json"),
                    "telemetry_sha256": _sha256(run.run_dir / "telemetry.csv"),
                }
                for run in runs
            ]
            for name, runs in all_runs.items()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "delay_candidates.json").write_text(
        json.dumps([_candidate_dict(item) for item in candidates], indent=2) + "\n"
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    params = {
        "model": "M0_dual_gain",
        "shared_delay_sec": best.delay_sec,
        "shared_delay_ms": best.delay_sec * 1000.0,
        "conditions": {
            condition.name: {
                "register_p": condition.register_p,
                "kp_eff": best.by_condition[condition.name].kp,
                "kd_eff": best.by_condition[condition.name].kd,
            }
            for condition in config.conditions
        },
    }
    (output / "params_m0_dual_gain.yaml").write_text(
        yaml.safe_dump(params, sort_keys=False, allow_unicode=True)
    )
    for condition in config.conditions:
        for run in all_runs[condition.name]:
            _save_plot(
                output / f"{condition.name}_{run.target_joint}_repeat{run.repeat_index}_{run.split_role}.png",
                run,
                fitted_results[(condition.name, run.run_dir)],
                condition.name,
                best,
            )
    report = [
        "# Jandi shared-delay dual-gain M0",
        "",
        f"- 공통 command delay: {best.delay_sec * 1000.0:.1f} ms",
    ]
    for condition in config.conditions:
        fit = best.by_condition[condition.name]
        report.append(
            f"- {condition.name}: Kp_eff={fit.kp:.6f}, Kd_eff={fit.kd:.6f}, "
            f"optimizer_success={fit.success}"
        )
    report.extend(
        [
            "",
            "## 해석 제한",
            "",
            "- repeat 1·2만 fit에 사용했고 repeat 3은 validation 전용입니다.",
            "- step edge 뒤 transient는 자기 plateau 기준으로 비교해 static hysteresis가 지연을 보상하지 않게 했습니다.",
            "- friction/backlash 자체는 아직 모델링하지 않았으므로 full-trajectory 정상상태 오차는 남을 수 있습니다.",
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n")
    return output

