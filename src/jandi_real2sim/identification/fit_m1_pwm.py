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

from jandi_real2sim.config import MUJOCO_DOF_ORDER, load_robot_config
from jandi_real2sim.identification.dataset import RunData, split_fit_validation
from jandi_real2sim.identification.fit_m0 import _huber
from jandi_real2sim.identification.fit_m0_dual_gain import (
    load_collection_campaign_runs,
)
from jandi_real2sim.identification.replay_pwm import (
    FixedBasePwmReplay,
    M1Parameters,
    PwmReplayConfig,
    PwmReplayResult,
)


PARAMETER_NAMES = (
    "drive_gain_nm_per_duty",
    "armature_kg_m2",
    "coulomb_friction_nm",
    "viscous_friction_nm_s_per_rad",
)


@dataclass(frozen=True)
class GainCondition:
    name: str
    register_p: int


@dataclass(frozen=True)
class PwmM1FitConfig:
    source: Path
    replay: PwmReplayConfig
    target_joints: tuple[str, str]
    pose_id: str
    conditions: tuple[GainCondition, GainCondition]
    initial_starts: tuple[M1Parameters, ...]
    bounds: tuple[tuple[float, float], ...]
    huber_delta_normalized: float
    velocity_loss_weight: float
    optimizer_maxiter: int
    optimizer_max_evaluations: int


@dataclass(frozen=True)
class M1Candidate:
    parameters: M1Parameters
    fit_loss: float
    success: bool
    evaluations: int
    start_index: int


def _parameters_from_mapping(values: dict[str, Any]) -> M1Parameters:
    return M1Parameters(**{name: float(values[name]) for name in PARAMETER_NAMES})


def _parameters_to_array(parameters: M1Parameters) -> np.ndarray:
    return np.asarray([getattr(parameters, name) for name in PARAMETER_NAMES])


def _parameters_from_array(values: np.ndarray) -> M1Parameters:
    return M1Parameters(*map(float, values))


def load_pwm_m1_fit_config(path: str | Path) -> PwmM1FitConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text())
    robot = load_robot_config(raw["robot_config"])
    target_joints = tuple(str(value) for value in raw["target_joints"])
    if len(target_joints) != 2 or any(name not in MUJOCO_DOF_ORDER for name in target_joints):
        raise ValueError("target_joints에는 유효한 두 관절이 필요합니다.")
    conditions = tuple(
        GainCondition(str(name), int(values["register_p"]))
        for name, values in raw["gain_conditions"].items()
    )
    if len(conditions) != 2:
        raise ValueError("gain_conditions는 P350/P850 두 조건이어야 합니다.")
    starts = tuple(_parameters_from_mapping(item) for item in raw["initial_starts"])
    if not starts:
        raise ValueError("initial_starts가 비어 있습니다.")
    bounds = tuple(tuple(map(float, raw["bounds"][name])) for name in PARAMETER_NAMES)
    for name, bound in zip(PARAMETER_NAMES, bounds):
        if len(bound) != 2 or bound[0] < 0.0 or bound[0] >= bound[1]:
            raise ValueError(f"{name} bounds가 잘못됐습니다: {bound}")
    for start in starts:
        for name, value, bound in zip(PARAMETER_NAMES, _parameters_to_array(start), bounds):
            if not bound[0] <= value <= bound[1]:
                raise ValueError(f"{name} 초기값 {value}가 bounds {bound} 밖입니다.")
    return PwmM1FitConfig(
        source=source,
        replay=PwmReplayConfig(
            model_xml=Path(raw["model_xml"]),
            robot=robot,
            physics_dt_sec=float(raw["physics_dt_sec"]),
            nominal_voltage_v=float(raw["nominal_voltage_v"]),
            torque_limit_nm=float(raw["torque_limit_nm"]),
        ),
        target_joints=(target_joints[0], target_joints[1]),
        pose_id=str(raw["pose_id"]),
        conditions=(conditions[0], conditions[1]),
        initial_starts=starts,
        bounds=bounds,
        huber_delta_normalized=float(raw["huber_delta_normalized"]),
        velocity_loss_weight=float(raw["velocity_loss_weight"]),
        optimizer_maxiter=int(raw["optimizer_maxiter"]),
        optimizer_max_evaluations=int(raw["optimizer_max_evaluations"]),
    )


def pwm_m1_loss(
    run: RunData,
    result: PwmReplayResult,
    config: PwmM1FitConfig,
) -> float:
    index = run.target_index
    mask = run.state_mask
    amplitude = max(
        float(np.max(np.abs(run.q_cmd_rad[:, index] - run.q_init_rad[index]))),
        0.01,
    )
    q_error = (result.q_rad[mask, index] - run.q_real_rad[mask, index]) / amplitude
    loss = float(np.mean(_huber(q_error, config.huber_delta_normalized)))
    if config.velocity_loss_weight > 0.0:
        real_dq = run.dq_real_rad_s[mask, index]
        dq_scale = max(float(np.nanpercentile(np.abs(real_dq), 95)), 0.05)
        dq_error = (result.dq_rad_s[mask, index] - real_dq) / dq_scale
        loss += config.velocity_loss_weight * float(
            np.mean(_huber(dq_error, config.huber_delta_normalized))
        )
    return loss


def _mean_loss(
    simulator: FixedBasePwmReplay,
    runs: tuple[RunData, ...],
    parameters: M1Parameters,
    config: PwmM1FitConfig,
) -> float:
    return float(
        np.mean(
            [
                pwm_m1_loss(run, simulator.run(run, parameters), config)
                for run in runs
            ]
        )
    )


def fit_pwm_m1_candidates(
    simulator: FixedBasePwmReplay,
    fit_runs: tuple[RunData, ...],
    config: PwmM1FitConfig,
) -> tuple[M1Candidate, ...]:
    candidates: list[M1Candidate] = []
    for start_index, start in enumerate(config.initial_starts, start=1):
        print(f"M1 start {start_index}/{len(config.initial_starts)}: {start}")

        def objective(values: np.ndarray) -> float:
            return _mean_loss(
                simulator,
                fit_runs,
                _parameters_from_array(values),
                config,
            )

        result = minimize(
            objective,
            x0=_parameters_to_array(start),
            method="Powell",
            bounds=config.bounds,
            options={
                "maxiter": config.optimizer_maxiter,
                "maxfev": config.optimizer_max_evaluations,
                "xtol": 1e-4,
                "ftol": 1e-6,
            },
        )
        candidate = M1Candidate(
            parameters=_parameters_from_array(result.x),
            fit_loss=float(result.fun),
            success=bool(result.success),
            evaluations=int(result.nfev),
            start_index=start_index,
        )
        candidates.append(candidate)
        print(
            f"  loss={candidate.fit_loss:.8f}, n={candidate.evaluations}, "
            f"success={candidate.success}, params={candidate.parameters}"
        )
    return tuple(candidates)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_metrics(
    run: RunData,
    result: PwmReplayResult,
    config: PwmM1FitConfig,
) -> dict[str, Any]:
    index = run.target_index
    mask = run.state_mask
    q_error = result.q_rad[mask, index] - run.q_real_rad[mask, index]
    dq_error = result.dq_rad_s[mask, index] - run.dq_real_rad_s[mask, index]
    return {
        "run_dir": str(run.run_dir),
        "joint": run.target_joint,
        "repeat_index": run.repeat_index,
        "split_role": run.split_role,
        "loss": pwm_m1_loss(run, result, config),
        "q_mae_rad": float(np.mean(np.abs(q_error))),
        "q_rmse_rad": float(np.sqrt(np.mean(q_error**2))),
        "dq_mae_rad_s": float(np.mean(np.abs(dq_error))),
        "max_abs_drive_torque_nm": float(
            np.max(np.abs(result.drive_torque_nm[:, index]))
        ),
        "torque_saturation_fraction": float(
            np.mean(
                np.abs(result.drive_torque_nm[:, index])
                >= config.replay.torque_limit_nm - 1e-9
            )
        ),
        "max_abs_pwm_duty": float(np.max(np.abs(result.pwm_duty[:, index]))),
        "sample_substep": result.sample_substep,
    }


def _save_plot(path: Path, run: RunData, result: PwmReplayResult, condition: str) -> None:
    index = run.target_index
    time = np.arange(len(run.q_cmd_rad)) / run.command_rate_hz
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(time, run.q_cmd_rad[:, index], "k--", lw=1.0, label="q_cmd")
    axes[0].plot(
        time[run.state_mask],
        run.q_real_rad[run.state_mask, index],
        lw=1.2,
        label="real",
    )
    axes[0].plot(time, result.q_rad[:, index], color="tab:red", lw=1.0, label="PWM M1")
    axes[0].set(ylabel="joint angle [rad]", title=f"{condition} PWM-input M1")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(time, result.pwm_duty[:, index], label="measured PWM duty")
    axes[1].plot(time, result.drive_torque_nm[:, index], label="equivalent drive torque [Nm]")
    axes[1].set(xlabel="time [s]", ylabel="input")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def identify_pwm_m1(
    campaigns: dict[str, Path],
    config: PwmM1FitConfig,
    output_root: str | Path,
) -> Path:
    all_runs: dict[str, tuple[RunData, ...]] = {}
    fit_runs: list[RunData] = []
    for condition in config.conditions:
        runs = load_collection_campaign_runs(
            campaigns[condition.name], config, condition.register_p
        )
        fit, _ = split_fit_validation(runs)
        all_runs[condition.name] = runs
        fit_runs.extend(fit)

    simulator = FixedBasePwmReplay(config.replay)
    candidates = fit_pwm_m1_candidates(simulator, tuple(fit_runs), config)
    best = min(candidates, key=lambda item: item.fit_loss)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(output_root).expanduser().resolve() / f"{timestamp}_m1_pwm"
    output.mkdir(parents=True, exist_ok=False)
    metrics: dict[str, Any] = {}
    results: dict[tuple[str, Path], PwmReplayResult] = {}
    for condition in config.conditions:
        condition_metrics = []
        for run in all_runs[condition.name]:
            result = simulator.run(run, best.parameters)
            results[(condition.name, run.run_dir)] = result
            condition_metrics.append(_run_metrics(run, result, config))
        metrics[condition.name] = condition_metrics
        for role in ("fit", "validation"):
            selected = [item for item in condition_metrics if item["split_role"] == role]
            metrics[f"{condition.name}_{role}_mean"] = {
                key: float(np.mean([item[key] for item in selected]))
                for key in ("loss", "q_mae_rad", "q_rmse_rad", "dq_mae_rad_s")
            }

    manifest = {
        "schema_version": 1,
        "method": "measured Present PWM input + shared output-side M1",
        "fit_config": str(config.source),
        "fit_config_sha256": _sha256(config.source),
        "robot_config": str(config.replay.robot.source),
        "robot_config_sha256": _sha256(config.replay.robot.source),
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
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "optimizer_candidates.json").write_text(
        json.dumps(
            [
                {
                    "start_index": item.start_index,
                    "fit_loss": item.fit_loss,
                    "success": item.success,
                    "evaluations": item.evaluations,
                    "parameters": {
                        name: getattr(item.parameters, name) for name in PARAMETER_NAMES
                    },
                }
                for item in candidates
            ],
            indent=2,
        )
        + "\n"
    )
    params = {
        "model": "M1_pwm_output_equivalent",
        "shared_across_position_p_registers": [item.name for item in config.conditions],
        "parameters": {
            name: getattr(best.parameters, name) for name in PARAMETER_NAMES
        },
        "nominal_voltage_v": config.replay.nominal_voltage_v,
        "torque_limit_nm": config.replay.torque_limit_nm,
    }
    (output / "params_m1_pwm.yaml").write_text(
        yaml.safe_dump(params, sort_keys=False, allow_unicode=True)
    )
    for condition in config.conditions:
        for run in all_runs[condition.name]:
            _save_plot(
                output
                / f"{condition.name}_{run.target_joint}_repeat{run.repeat_index}_{run.split_role}.png",
                run,
                results[(condition.name, run.run_dir)],
                condition.name,
            )

    validation_losses = [
        item["loss"]
        for condition in config.conditions
        for item in metrics[condition.name]
        if item["split_role"] == "validation"
    ]
    report = [
        "# Jandi measured-PWM M1 identification",
        "",
        "## 선택된 출력축 등가 파라미터",
        "",
        *[f"- {name}: {getattr(best.parameters, name):.9g}" for name in PARAMETER_NAMES],
        f"- fit loss: {best.fit_loss:.9g}",
        f"- validation loss mean: {float(np.mean(validation_losses)):.9g}",
        "",
        "## 모델 계약",
        "",
        "- Position P/D와 command delay는 피팅하지 않고 실측 Present PWM을 입력으로 사용했습니다.",
        "- drive gain은 motor constant·기어비·효율을 합친 출력축 등가값입니다.",
        "- Coulomb/viscous/armature는 12개 MX-106에 공통 적용했습니다.",
        "- repeat 1·2만 fit, repeat 3은 validation 전용입니다.",
        "- backlash, Stribeck, load-dependent friction은 아직 포함하지 않았습니다.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    return output
