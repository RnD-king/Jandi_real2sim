from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import minimize

from jandi_real2sim.identification.dataset import RunData, load_run, split_fit_validation
from jandi_real2sim.identification.fit_m1_pwm import (
    PARAMETER_NAMES,
    PwmM1FitConfig,
    _run_metrics,
    _save_plot,
    load_pwm_m1_fit_config,
    pwm_m1_loss,
)
from jandi_real2sim.identification.replay_pwm import (
    FixedBasePwmReplay,
    M1Parameters,
)


TRAJECTORY_SPECS = {
    "step": ("multi_amplitude_step", "compact_step"),
    "triangle": ("slow_triangle", "triangle"),
    "multisine": ("policy_band_multisine", "multisine"),
}


@dataclass(frozen=True)
class StageSettings:
    starts: tuple[M1Parameters, ...]
    maxiter: int
    max_evaluations: int


@dataclass(frozen=True)
class StagedM1FitConfig:
    source: Path
    base: PwmM1FitConfig
    triangle: StageSettings
    multisine: StageSettings
    final: StageSettings
    refinement_fraction: float
    refinement_min_global_fraction: float


@dataclass(frozen=True)
class StageCandidate:
    stage: str
    active_parameters: tuple[str, ...]
    parameters: M1Parameters
    loss: float
    success: bool
    evaluations: int
    start_index: int


def _parameter_mapping(parameters: M1Parameters) -> dict[str, float]:
    return {name: float(getattr(parameters, name)) for name in PARAMETER_NAMES}


def _parameters(mapping: dict[str, Any]) -> M1Parameters:
    return M1Parameters(**{name: float(mapping[name]) for name in PARAMETER_NAMES})


def _stage_settings(raw: dict[str, Any]) -> StageSettings:
    starts = tuple(_parameters(item) for item in raw["starts"])
    if not starts:
        raise ValueError("각 stage에는 시작점이 최소 하나 필요합니다.")
    return StageSettings(
        starts=starts,
        maxiter=int(raw["maxiter"]),
        max_evaluations=int(raw["max_evaluations"]),
    )


def load_staged_m1_fit_config(path: str | Path) -> StagedM1FitConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text())
    base = load_pwm_m1_fit_config(source)
    stages = raw["stages"]
    refinement_fraction = float(raw["final_refinement_fraction"])
    minimum_fraction = float(raw["final_min_global_span_fraction"])
    if not 0.0 < refinement_fraction <= 1.0:
        raise ValueError("final_refinement_fraction은 (0,1]이어야 합니다.")
    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError("final_min_global_span_fraction은 [0,1]이어야 합니다.")
    return StagedM1FitConfig(
        source=source,
        base=base,
        triangle=_stage_settings(stages["triangle"]),
        multisine=_stage_settings(stages["multisine"]),
        final=_stage_settings(stages["final"]),
        refinement_fraction=refinement_fraction,
        refinement_min_global_fraction=minimum_fraction,
    )


def load_named_trajectory_runs(
    campaign_root: str | Path,
    config: PwmM1FitConfig,
    *,
    register_p: int,
    trajectory: str,
) -> tuple[RunData, ...]:
    if trajectory not in TRAJECTORY_SPECS:
        raise ValueError(f"알 수 없는 trajectory: {trajectory}")
    experiment_name, experiment_type = TRAJECTORY_SPECS[trajectory]
    root = Path(campaign_root).expanduser().resolve()
    completed = json.loads((root / "campaign_status.json").read_text())["completed"]
    runs: list[RunData] = []
    for repeat in (1, 2, 3):
        expected_role = "fit" if repeat in (1, 2) else "validation"
        status_role = "diagnostic" if trajectory == "triangle" else expected_role
        for joint in config.target_joints:
            key = f"{experiment_name}/{joint}/{repeat}/{status_role}"
            if key not in completed:
                raise ValueError(f"campaign status에 run이 없습니다: {key}")
            run = load_run(root / "runs" / completed[key])
            if run.metadata.get("experiment_type") != experiment_type:
                raise ValueError(f"{run.run_dir}: experiment_type 불일치")
            if run.metadata.get("campaign_experiment_name") != experiment_name:
                raise ValueError(f"{run.run_dir}: campaign experiment 이름 불일치")
            if run.metadata.get("pose_id") != config.pose_id:
                raise ValueError(f"{run.run_dir}: pose_id 불일치")
            actual_p = {
                int(values["position_p_gain"])
                for values in run.metadata.get("actuator_settings", {}).values()
            }
            if actual_p != {register_p}:
                raise ValueError(
                    f"{run.run_dir}: Position P={actual_p}, expected={register_p}"
                )
            run = dataclasses.replace(
                run,
                repeat_index=repeat,
                split_role=expected_role,
            )
            runs.append(run)
    split_fit_validation(tuple(runs))
    return tuple(runs)


def load_all_trajectory_runs(
    campaigns: dict[str, Path], config: StagedM1FitConfig
) -> dict[str, dict[str, tuple[RunData, ...]]]:
    result: dict[str, dict[str, tuple[RunData, ...]]] = {}
    for condition in config.base.conditions:
        result[condition.name] = {
            trajectory: load_named_trajectory_runs(
                campaigns[condition.name],
                config.base,
                register_p=condition.register_p,
                trajectory=trajectory,
            )
            for trajectory in TRAJECTORY_SPECS
        }
    return result


def _merge_parameters(
    fixed: M1Parameters, active_names: tuple[str, ...], active_values: np.ndarray
) -> M1Parameters:
    mapping = _parameter_mapping(fixed)
    mapping.update(zip(active_names, map(float, active_values)))
    return _parameters(mapping)


def _active_bounds(
    config: PwmM1FitConfig, active_names: tuple[str, ...]
) -> tuple[tuple[float, float], ...]:
    by_name = dict(zip(PARAMETER_NAMES, config.bounds))
    return tuple(by_name[name] for name in active_names)


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


def optimize_stage(
    *,
    name: str,
    simulator: FixedBasePwmReplay,
    runs: tuple[RunData, ...],
    active_names: tuple[str, ...],
    starts: tuple[M1Parameters, ...],
    bounds: tuple[tuple[float, float], ...],
    maxiter: int,
    max_evaluations: int,
    loss_config: PwmM1FitConfig,
) -> tuple[StageCandidate, tuple[StageCandidate, ...]]:
    candidates: list[StageCandidate] = []
    print(f"\n[{name}] runs={len(runs)}, active={', '.join(active_names)}")
    for start_index, start in enumerate(starts, start=1):
        x0 = np.asarray([getattr(start, parameter) for parameter in active_names])
        print(f"  start {start_index}/{len(starts)}: {start}")

        def objective(values: np.ndarray) -> float:
            return _mean_loss(
                simulator,
                runs,
                _merge_parameters(start, active_names, values),
                loss_config,
            )

        result = minimize(
            objective,
            x0=x0,
            method="Powell",
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "maxfev": max_evaluations,
                "xtol": 1e-4,
                "ftol": 1e-6,
            },
        )
        candidate = StageCandidate(
            stage=name,
            active_parameters=active_names,
            parameters=_merge_parameters(start, active_names, result.x),
            loss=float(result.fun),
            success=bool(result.success),
            evaluations=int(result.nfev),
            start_index=start_index,
        )
        candidates.append(candidate)
        print(
            f"    loss={candidate.loss:.8f}, n={candidate.evaluations}, "
            f"success={candidate.success}, params={candidate.parameters}"
        )
    best = min(candidates, key=lambda item: item.loss)
    print(f"  best start={best.start_index}, loss={best.loss:.8f}")
    return best, tuple(candidates)


def _replace_stage_starts(
    templates: tuple[M1Parameters, ...],
    previous: M1Parameters,
    keep_from_previous: tuple[str, ...],
) -> tuple[M1Parameters, ...]:
    starts = []
    for template in templates:
        mapping = _parameter_mapping(template)
        for name in keep_from_previous:
            mapping[name] = getattr(previous, name)
        starts.append(_parameters(mapping))
    return tuple(starts)


def _refinement_bounds(
    center: M1Parameters, config: StagedM1FitConfig
) -> tuple[tuple[float, float], ...]:
    bounds = []
    for name, global_bound in zip(PARAMETER_NAMES, config.base.bounds):
        value = getattr(center, name)
        global_width = global_bound[1] - global_bound[0]
        half_span = max(
            abs(value) * config.refinement_fraction,
            0.5 * global_width * config.refinement_min_global_fraction,
        )
        bounds.append(
            (max(global_bound[0], value - half_span), min(global_bound[1], value + half_span))
        )
    return tuple(bounds)


def _flatten_runs(
    all_runs: dict[str, dict[str, tuple[RunData, ...]]],
    trajectories: tuple[str, ...],
    role: str,
) -> tuple[RunData, ...]:
    return tuple(
        run
        for condition_runs in all_runs.values()
        for trajectory in trajectories
        for run in condition_runs[trajectory]
        if run.split_role == role
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identify_staged_pwm_m1(
    campaigns: dict[str, Path],
    config: StagedM1FitConfig,
    output_root: str | Path,
) -> Path:
    all_runs = load_all_trajectory_runs(campaigns, config)
    simulator = FixedBasePwmReplay(config.base.replay)

    triangle_runs = _flatten_runs(all_runs, ("triangle",), "fit")
    triangle_best, triangle_candidates = optimize_stage(
        name="triangle_friction",
        simulator=simulator,
        runs=triangle_runs,
        active_names=("drive_gain_nm_per_duty", "coulomb_friction_nm"),
        starts=config.triangle.starts,
        bounds=_active_bounds(
            config.base, ("drive_gain_nm_per_duty", "coulomb_friction_nm")
        ),
        maxiter=config.triangle.maxiter,
        max_evaluations=config.triangle.max_evaluations,
        loss_config=config.base,
    )

    multisine_starts = _replace_stage_starts(
        config.multisine.starts,
        triangle_best.parameters,
        ("drive_gain_nm_per_duty", "coulomb_friction_nm"),
    )
    multisine_runs = _flatten_runs(all_runs, ("multisine",), "fit")
    multisine_best, multisine_candidates = optimize_stage(
        name="multisine_dynamics",
        simulator=simulator,
        runs=multisine_runs,
        active_names=(
            "drive_gain_nm_per_duty",
            "armature_kg_m2",
            "viscous_friction_nm_s_per_rad",
        ),
        starts=multisine_starts,
        bounds=_active_bounds(
            config.base,
            (
                "drive_gain_nm_per_duty",
                "armature_kg_m2",
                "viscous_friction_nm_s_per_rad",
            ),
        ),
        maxiter=config.multisine.maxiter,
        max_evaluations=config.multisine.max_evaluations,
        loss_config=config.base,
    )

    final_starts = _replace_stage_starts(
        config.final.starts,
        multisine_best.parameters,
        PARAMETER_NAMES,
    )
    # 첫 시작점은 단계식 최적값, 둘째부터는 YAML scale을 곱해 좁은 영역의
    # 시작점 의존성을 점검한다.
    adjusted_final_starts = [final_starts[0]]
    for template in config.final.starts[1:]:
        mapping = {
            name: getattr(multisine_best.parameters, name) * getattr(template, name)
            for name in PARAMETER_NAMES
        }
        adjusted_final_starts.append(_parameters(mapping))
    final_bounds = _refinement_bounds(multisine_best.parameters, config)
    clipped_starts = tuple(
        _parameters(
            {
                name: float(np.clip(getattr(start, name), *bound))
                for name, bound in zip(PARAMETER_NAMES, final_bounds)
            }
        )
        for start in adjusted_final_starts
    )
    final_runs = _flatten_runs(all_runs, ("step", "triangle", "multisine"), "fit")
    final_best, final_candidates = optimize_stage(
        name="joint_refinement",
        simulator=simulator,
        runs=final_runs,
        active_names=PARAMETER_NAMES,
        starts=clipped_starts,
        bounds=final_bounds,
        maxiter=config.final.maxiter,
        max_evaluations=config.final.max_evaluations,
        loss_config=config.base,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(output_root).expanduser().resolve() / f"{timestamp}_m1_pwm_staged"
    output.mkdir(parents=True, exist_ok=False)

    metrics: dict[str, Any] = {}
    for condition, condition_runs in all_runs.items():
        metrics[condition] = {}
        for trajectory, runs in condition_runs.items():
            items = []
            for run in runs:
                result = simulator.run(run, final_best.parameters)
                items.append(_run_metrics(run, result, config.base))
                _save_plot(
                    output
                    / f"{condition}_{trajectory}_{run.target_joint}_repeat{run.repeat_index}_{run.split_role}.png",
                    run,
                    result,
                    f"{condition} {trajectory}",
                )
            metrics[condition][trajectory] = items
            for role in ("fit", "validation"):
                selected = [item for item in items if item["split_role"] == role]
                metrics[condition][f"{trajectory}_{role}_mean"] = {
                    key: float(np.mean([item[key] for item in selected]))
                    for key in ("loss", "q_mae_rad", "q_rmse_rad", "dq_mae_rad_s")
                }

    candidates = triangle_candidates + multisine_candidates + final_candidates
    candidate_json = [
        {
            "stage": item.stage,
            "start_index": item.start_index,
            "active_parameters": item.active_parameters,
            "loss": item.loss,
            "success": item.success,
            "evaluations": item.evaluations,
            "parameters": _parameter_mapping(item.parameters),
        }
        for item in candidates
    ]
    (output / "optimizer_stages.json").write_text(
        json.dumps(candidate_json, indent=2) + "\n"
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    params = {
        "model": "M1_pwm_output_equivalent_staged",
        "parameters": _parameter_mapping(final_best.parameters),
        "stages": {
            "triangle": _parameter_mapping(triangle_best.parameters),
            "multisine": _parameter_mapping(multisine_best.parameters),
            "joint_refinement": _parameter_mapping(final_best.parameters),
        },
        "nominal_voltage_v": config.base.replay.nominal_voltage_v,
        "torque_limit_nm": config.base.replay.torque_limit_nm,
    }
    (output / "params_m1_pwm_staged.yaml").write_text(
        yaml.safe_dump(params, sort_keys=False, allow_unicode=True)
    )
    manifest = {
        "schema_version": 1,
        "method": "triangle -> multisine -> step/triangle/multisine joint refinement",
        "fit_config": str(config.source),
        "fit_config_sha256": _sha256(config.source),
        "campaigns": {name: str(path.expanduser().resolve()) for name, path in campaigns.items()},
        "runs": [
            {
                "condition": condition,
                "trajectory": trajectory,
                "run_dir": str(run.run_dir),
                "metadata_sha256": _sha256(run.run_dir / "metadata.json"),
                "telemetry_sha256": _sha256(run.run_dir / "telemetry.csv"),
            }
            for condition, condition_runs in all_runs.items()
            for trajectory, runs in condition_runs.items()
            for run in runs
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    validation_losses = [
        item["loss"]
        for condition_data in metrics.values()
        for trajectory in TRAJECTORY_SPECS
        for item in condition_data[trajectory]
        if item["split_role"] == "validation"
    ]
    report = [
        "# Jandi staged measured-PWM M1 identification",
        "",
        "## Final parameters",
        "",
        *[
            f"- {name}: {getattr(final_best.parameters, name):.9g}"
            for name in PARAMETER_NAMES
        ],
        f"- joint-refinement fit loss: {final_best.loss:.9g}",
        f"- all-trajectory validation loss mean: {float(np.mean(validation_losses)):.9g}",
        "",
        "## Stage contract",
        "",
        "- Triangle: drive gain + Coulomb friction",
        "- Multisine: drive gain + armature + viscous friction; Coulomb fixed",
        "- Joint refinement: all four parameters in narrowed bounds",
        "- repeat/seed 1·2 fit, 3 validation",
        "- backlash/Stribeck/load-dependent friction are not included yet",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    return output
