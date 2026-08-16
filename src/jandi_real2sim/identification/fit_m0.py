from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import minimize

from jandi_real2sim.config import MUJOCO_DOF_ORDER
from jandi_real2sim.identification.dataset import (
    RunData,
    estimate_delay_samples,
    split_fit_validation,
)
from jandi_real2sim.identification.replay import (
    FixedBaseReplay,
    ReplayConfig,
    ReplayResult,
)


@dataclass(frozen=True)
class FitConfig:
    source: Path
    replay: ReplayConfig
    target_joints: tuple[str, str]
    pose_id: str
    initial_kp: float
    initial_kd: float
    kp_bounds: tuple[float, float]
    kd_bounds: tuple[float, float]
    delay_values_sec: np.ndarray
    huber_delta_normalized: float
    velocity_loss_weight: float
    optimizer_maxiter: int
    optimizer_max_evaluations: int
    delay_shortlist_count: int
    delay_refine_radius_steps: int


@dataclass(frozen=True)
class Candidate:
    delay_sec: float
    kp: float
    kd: float
    fit_loss: float
    success: bool
    evaluations: int


def load_fit_config(path: str | Path) -> FitConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text())
    kp_baseline = np.asarray(raw["kp_baseline"], dtype=np.float64)
    kd_baseline = np.asarray(raw["kd_baseline"], dtype=np.float64)
    if kp_baseline.shape != (12,) or kd_baseline.shape != (12,):
        raise ValueError("kp_baseline/kd_baseline은 MuJoCo 순서의 12개 값이어야 합니다.")
    target_joints = tuple(str(value) for value in raw["target_joints"])
    if len(target_joints) != 2 or any(
        joint not in MUJOCO_DOF_ORDER for joint in target_joints
    ):
        raise ValueError("target_joints에는 유효한 두 관절이 필요합니다.")
    delay_min = float(raw["delay_min_sec"])
    delay_max = float(raw["delay_max_sec"])
    delay_step = float(raw["delay_step_sec"])
    delay_values = np.arange(
        delay_min, delay_max + 0.5 * delay_step, delay_step, dtype=np.float64
    )
    return FitConfig(
        source=source,
        replay=ReplayConfig(
            model_xml=Path(raw["model_xml"]),
            physics_dt_sec=float(raw["physics_dt_sec"]),
            torque_limit_nm=float(raw["torque_limit_nm"]),
            kp_baseline=kp_baseline,
            kd_baseline=kd_baseline,
        ),
        target_joints=(target_joints[0], target_joints[1]),
        pose_id=str(raw["pose_id"]),
        initial_kp=float(raw["initial_kp"]),
        initial_kd=float(raw["initial_kd"]),
        kp_bounds=tuple(map(float, raw["kp_bounds"])),
        kd_bounds=tuple(map(float, raw["kd_bounds"])),
        delay_values_sec=delay_values,
        huber_delta_normalized=float(raw["huber_delta_normalized"]),
        velocity_loss_weight=float(raw["velocity_loss_weight"]),
        optimizer_maxiter=int(raw["optimizer_maxiter"]),
        optimizer_max_evaluations=int(raw["optimizer_max_evaluations"]),
        delay_shortlist_count=int(raw["delay_shortlist_count"]),
        delay_refine_radius_steps=int(raw["delay_refine_radius_steps"]),
    )


def _normalizers(run: RunData) -> tuple[float, float]:
    index = run.target_index
    command = run.q_cmd_rad[:, index]
    center = float(run.metadata["pose_rad"][run.target_joint])
    q_scale = max(float(np.max(np.abs(command - center))), 1e-3)
    dq_scale = max(q_scale * run.command_rate_hz, 1e-3)
    return q_scale, dq_scale


def _huber(values: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(values)
    return np.where(
        absolute <= delta,
        0.5 * values**2,
        delta * (absolute - 0.5 * delta),
    )


def replay_loss(
    run: RunData,
    result: ReplayResult,
    *,
    huber_delta_normalized: float,
    velocity_loss_weight: float,
) -> float:
    index = run.target_index
    mask = run.state_mask
    q_scale, dq_scale = _normalizers(run)
    q_error = (result.q_rad[mask, index] - run.q_real_rad[mask, index]) / q_scale
    loss = float(np.mean(_huber(q_error, huber_delta_normalized)))
    if velocity_loss_weight > 0.0:
        dq_error = (
            result.dq_rad_s[mask, index] - run.dq_real_rad_s[mask, index]
        ) / dq_scale
        loss += velocity_loss_weight * float(
            np.mean(_huber(dq_error, huber_delta_normalized))
        )
    return loss


def _evaluate(
    simulator: FixedBaseReplay,
    runs: Iterable[RunData],
    config: FitConfig,
    delay_sec: float,
    kp: float,
    kd: float,
) -> tuple[float, list[ReplayResult]]:
    results = [
        simulator.run(run, delay_sec=delay_sec, target_kp=kp, target_kd=kd)
        for run in runs
    ]
    losses = [
        replay_loss(
            run,
            result,
            huber_delta_normalized=config.huber_delta_normalized,
            velocity_loss_weight=config.velocity_loss_weight,
        )
        for run, result in zip(runs, results)
    ]
    return float(np.mean(losses)), results


def _shortlist_delay_values(
    screened: tuple[Candidate, ...],
    all_delay_values: np.ndarray,
    shortlist_count: int,
    radius_steps: int,
) -> tuple[float, ...]:
    if shortlist_count < 1 or radius_steps < 0:
        raise ValueError("delay shortlist count는 1 이상, radius는 0 이상이어야 합니다.")
    ranked = sorted(screened, key=lambda candidate: candidate.fit_loss)
    selected_indices: set[int] = set()
    for candidate in ranked[:shortlist_count]:
        center = int(np.argmin(np.abs(all_delay_values - candidate.delay_sec)))
        for index in range(center - radius_steps, center + radius_steps + 1):
            if 0 <= index < len(all_delay_values):
                selected_indices.add(index)
    return tuple(float(all_delay_values[index]) for index in sorted(selected_indices))


def fit_candidates(
    simulator: FixedBaseReplay,
    fit_runs: tuple[RunData, ...],
    config: FitConfig,
) -> tuple[Candidate, tuple[Candidate, ...], tuple[Candidate, ...]]:
    # 1단계: 현재 PD에서 모든 delay를 1회씩만 재생한다. 기존 구현처럼
    # 26개 delay마다 Powell을 완전히 돌리지 않는다.
    screened: list[Candidate] = []
    for delay_sec in config.delay_values_sec:
        loss, _ = _evaluate(
            simulator,
            fit_runs,
            config,
            float(delay_sec),
            config.initial_kp,
            config.initial_kd,
        )
        candidate = Candidate(
            delay_sec=float(delay_sec),
            kp=config.initial_kp,
            kd=config.initial_kd,
            fit_loss=loss,
            success=True,
            evaluations=1,
        )
        screened.append(candidate)
        print(
            f"screen delay={candidate.delay_sec * 1000:5.1f} ms  "
            f"loss={candidate.fit_loss:.8f}"
        )

    selected_delays = _shortlist_delay_values(
        tuple(screened),
        config.delay_values_sec,
        config.delay_shortlist_count,
        config.delay_refine_radius_steps,
    )
    print(
        "refine delays: "
        + ", ".join(f"{delay * 1000.0:.1f} ms" for delay in selected_delays)
    )

    # 2단계: screening 상위 지연과 바로 이웃에서만 Kp/Kd를 정밀화한다.
    candidates: list[Candidate] = []
    for delay_sec in selected_delays:
        def objective(parameters: np.ndarray) -> float:
            return _evaluate(
                simulator,
                fit_runs,
                config,
                float(delay_sec),
                float(parameters[0]),
                float(parameters[1]),
            )[0]

        result = minimize(
            objective,
            x0=np.asarray([config.initial_kp, config.initial_kd]),
            method="Powell",
            bounds=(config.kp_bounds, config.kd_bounds),
            options={
                "maxiter": config.optimizer_maxiter,
                "maxfev": config.optimizer_max_evaluations,
                "xtol": 1e-3,
                "ftol": 1e-5,
            },
        )
        candidate = Candidate(
            delay_sec=float(delay_sec),
            kp=float(result.x[0]),
            kd=float(result.x[1]),
            fit_loss=float(result.fun),
            success=bool(result.success),
            evaluations=int(result.nfev),
        )
        candidates.append(candidate)
        print(
            f"refine delay={candidate.delay_sec * 1000:5.1f} ms  "
            f"Kp={candidate.kp:8.4f}  Kd={candidate.kd:8.4f}  "
            f"fit_loss={candidate.fit_loss:.8f}"
        )
    best = min(candidates, key=lambda candidate: candidate.fit_loss)
    return best, tuple(candidates), tuple(screened)


def _run_metrics(
    run: RunData, result: ReplayResult, torque_limit_nm: float
) -> dict[str, Any]:
    index = run.target_index
    mask = run.state_mask
    error = result.q_rad[mask, index] - run.q_real_rad[mask, index]
    q_scale, _ = _normalizers(run)
    return {
        "run_dir": str(run.run_dir),
        "joint": run.target_joint,
        "repeat_index": run.repeat_index,
        "split_role": run.split_role,
        "mae_rad": float(np.mean(np.abs(error))),
        "rmse_rad": float(np.sqrt(np.mean(error**2))),
        "nrmse_by_amplitude": float(np.sqrt(np.mean(error**2)) / q_scale),
        "max_abs_error_rad": float(np.max(np.abs(error))),
        "max_abs_tau_nm": float(np.max(np.abs(result.tau_nm[:, index]))),
        "torque_saturation_fraction": float(
            np.mean(np.abs(result.tau_nm[:, index]) >= torque_limit_nm - 1e-9)
        ),
        "sample_substep": result.sample_substep,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _serializable_candidate(candidate: Candidate) -> dict[str, Any]:
    return {
        "delay_sec": candidate.delay_sec,
        "delay_ms": candidate.delay_sec * 1000.0,
        "kp_eff": candidate.kp,
        "kd_eff": candidate.kd,
        "fit_loss": candidate.fit_loss,
        "success": candidate.success,
        "evaluations": candidate.evaluations,
    }


def _save_plot(
    output: Path,
    run: RunData,
    baseline: ReplayResult,
    fitted: ReplayResult,
) -> None:
    index = run.target_index
    time = np.arange(len(run.q_cmd_rad)) / run.command_rate_hz
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(time, run.q_cmd_rad[:, index], "k--", lw=1.0, label="q_cmd")
    axis.plot(
        time[run.state_mask],
        run.q_real_rad[run.state_mask, index],
        color="tab:blue",
        lw=1.2,
        label="real",
    )
    axis.plot(time, baseline.q_rad[:, index], color="0.6", lw=1.0, label="baseline")
    axis.plot(time, fitted.q_rad[:, index], color="tab:red", lw=1.0, label="M0 fitted")
    axis.set(xlabel="time [s]", ylabel="joint angle [rad]")
    axis.grid(alpha=0.25)
    axis.legend(ncol=4)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def identify_m0(
    runs: tuple[RunData, ...],
    config: FitConfig,
    output_root: str | Path,
    *,
    campaign_id: str | None = None,
    campaign_source: str | Path | None = None,
) -> Path:
    fit_runs, validation_runs = split_fit_validation(runs)
    simulator = FixedBaseReplay(config.replay)
    baseline_parameters = Candidate(
        delay_sec=0.0,
        kp=config.initial_kp,
        kd=config.initial_kd,
        fit_loss=float("nan"),
        success=True,
        evaluations=0,
    )
    best, candidates, screened = fit_candidates(simulator, fit_runs, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(output_root).expanduser().resolve() / f"{timestamp}_m0_ankle_roll"
    output.mkdir(parents=True, exist_ok=False)

    baseline_results = {
        run.run_dir: simulator.run(
            run,
            delay_sec=baseline_parameters.delay_sec,
            target_kp=baseline_parameters.kp,
            target_kd=baseline_parameters.kd,
        )
        for run in runs
    }
    fitted_results = {
        run.run_dir: simulator.run(
            run, delay_sec=best.delay_sec, target_kp=best.kp, target_kd=best.kd
        )
        for run in runs
    }
    metrics = {
        "baseline": [
            _run_metrics(run, baseline_results[run.run_dir], config.replay.torque_limit_nm)
            for run in runs
        ],
        "fitted": [
            _run_metrics(run, fitted_results[run.run_dir], config.replay.torque_limit_nm)
            for run in runs
        ],
    }
    for label in ("baseline", "fitted"):
        for role in ("fit", "validation"):
            selected = [item for item in metrics[label] if item["split_role"] == role]
            metrics[f"{label}_{role}_mean"] = {
                key: float(np.mean([item[key] for item in selected]))
                for key in ("mae_rad", "rmse_rad", "nrmse_by_amplitude")
            }

    manifest = {
        "schema_version": 1,
        "method": "M0 common delay + shared RL6/LL6 effective PD",
        "campaign_id": campaign_id,
        "campaign_source": (
            str(Path(campaign_source).expanduser().resolve())
            if campaign_source is not None
            else None
        ),
        "campaign_sha256": (
            _sha256(Path(campaign_source).expanduser().resolve())
            if campaign_source is not None
            else None
        ),
        "fit_config": str(config.source),
        "fit_config_sha256": _sha256(config.source),
        "model_xml": str(config.replay.model_xml.expanduser().resolve()),
        "model_xml_sha256": _sha256(config.replay.model_xml.expanduser().resolve()),
        "runs": [
            {
                "run_dir": str(run.run_dir),
                "joint": run.target_joint,
                "repeat_index": run.repeat_index,
                "split_role": run.split_role,
                "metadata_sha256": _sha256(run.run_dir / "metadata.json"),
                "telemetry_sha256": _sha256(run.run_dir / "telemetry.csv"),
            }
            for run in runs
        ],
    }
    threshold_delays = {
        run.run_dir.name: estimate_delay_samples(run) for run in runs
    }
    params = {
        "model": "M0",
        "target_joints": list(config.target_joints),
        "shared": _serializable_candidate(best),
        "torque_limit_nm": config.replay.torque_limit_nm,
        "physics_dt_sec": config.replay.physics_dt_sec,
        "command_rate_hz": runs[0].command_rate_hz,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "params_m0.yaml").write_text(
        yaml.safe_dump(params, sort_keys=False, allow_unicode=True)
    )
    (output / "delay_candidates.json").write_text(
        json.dumps(
            {
                "threshold_diagnostic_command_samples": threshold_delays,
                "screening_at_initial_pd": [
                    _serializable_candidate(item) for item in screened
                ],
                "refined": [_serializable_candidate(item) for item in candidates],
            },
            indent=2,
        )
        + "\n"
    )
    for run in runs:
        _save_plot(
            output / f"{run.target_joint}_repeat{run.repeat_index}_{run.split_role}.png",
            run,
            baseline_results[run.run_dir],
            fitted_results[run.run_dir],
        )

    bounds_warning = []
    if abs(best.kp - config.kp_bounds[0]) < 0.02 * np.ptp(config.kp_bounds) or abs(
        best.kp - config.kp_bounds[1]
    ) < 0.02 * np.ptp(config.kp_bounds):
        bounds_warning.append("Kp가 탐색 경계 부근입니다.")
    if abs(best.kd - config.kd_bounds[0]) < 0.02 * np.ptp(config.kd_bounds) or abs(
        best.kd - config.kd_bounds[1]
    ) < 0.02 * np.ptp(config.kd_bounds):
        bounds_warning.append("Kd가 탐색 경계 부근입니다.")
    report = [
        "# Jandi M0 ankle-roll identification",
        "",
        f"- 공통 지연: {best.delay_sec * 1000.0:.1f} ms",
        f"- 공통 Kp_eff: {best.kp:.6f}",
        f"- 공통 Kd_eff: {best.kd:.6f}",
        f"- fit NRMSE: {metrics['fitted_fit_mean']['nrmse_by_amplitude']:.6f}",
        f"- validation NRMSE: {metrics['fitted_validation_mean']['nrmse_by_amplitude']:.6f}",
        f"- baseline validation NRMSE: {metrics['baseline_validation_mean']['nrmse_by_amplitude']:.6f}",
        "",
        "## 판정 주의",
        "",
        "- 반복 1·2만 최적화에 사용했고 반복 3은 완전히 분리해 검증했습니다.",
        "- 이 값은 MX-106 레지스터 게인이 아니라 현재 고정베이스 MuJoCo 모델의 유효 PD입니다.",
        "- M0 잔차에 히스테리시스나 방향 의존성이 남으면 다음 M1에서 마찰·백래시를 추가합니다.",
    ]
    report.extend(f"- {warning}" for warning in bounds_warning)
    (output / "report.md").write_text("\n".join(report) + "\n")
    return output
