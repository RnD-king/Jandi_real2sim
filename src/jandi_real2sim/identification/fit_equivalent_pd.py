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
from jandi_real2sim.identification.dataset import RunData, load_run, split_fit_validation
from jandi_real2sim.identification.fit_m0 import _huber
from jandi_real2sim.identification.fit_m0_dual_gain import (
    GainCondition,
    load_collection_campaign_runs,
)
from jandi_real2sim.identification.replay_equivalent_pd import (
    EquivalentParameters,
    EquivalentReplayConfig,
    EquivalentReplayResult,
    FixedBaseEquivalentPdReplay,
)


@dataclass(frozen=True)
class EquivalentFitConfig:
    source: Path
    replay: EquivalentReplayConfig
    target_joints: tuple[str, str]
    pose_id: str
    conditions: tuple[GainCondition, GainCondition]
    kp_per_register_bounds: tuple[float, float]
    initial_kp_per_register: float
    fixed_kd_eff: float
    backlash_bounds_rad: tuple[float, float]
    coulomb_bounds_nm: tuple[float, float]
    delay_values_sec: np.ndarray
    delay_refine_count: int
    transient_window_sec: float
    plateau_tail_sec: float
    huber_delta_normalized: float
    velocity_loss_weight: float
    plateau_loss_weight: float
    stage1_max_evaluations: int
    stage2_max_evaluations: int
    final_max_evaluations: int
    final_starts: int


@dataclass(frozen=True)
class ConditionPd:
    kp: float
    kd: float


@dataclass(frozen=True)
class EquivalentCandidate:
    delay_sec: float
    by_condition: dict[str, ConditionPd]
    backlash_total_rad: float
    coulomb_friction_nm: float
    loss: float
    success: bool
    evaluations: int
    kp_per_register: float
    shared_kd: float


def _pair(raw: dict[str, Any], name: str) -> tuple[float, float]:
    values = tuple(map(float, raw[name]))
    if len(values) != 2 or values[0] < 0.0 or values[0] >= values[1]:
        raise ValueError(f"{name} 범위가 잘못됐습니다: {values}")
    return values


def load_equivalent_fit_config(path: str | Path) -> EquivalentFitConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text())
    targets = tuple(map(str, raw["target_joints"]))
    if len(targets) != 2 or any(name not in MUJOCO_DOF_ORDER for name in targets):
        raise ValueError("target_joints에는 유효한 두 관절이 필요합니다.")
    initial_scale = float(raw["initial_kp_per_register"])
    fixed_kd = float(raw["fixed_kd_eff"])
    conditions = tuple(
        GainCondition(
            name=str(name),
            register_p=int(values["register_p"]),
            initial_kp=initial_scale * int(values["register_p"]),
            initial_kd=fixed_kd,
        )
        for name, values in raw["gain_conditions"].items()
    )
    if len(conditions) != 2:
        raise ValueError("gain_conditions는 정확히 두 조건이어야 합니다.")
    dt = float(raw["physics_dt_sec"])
    delay_step = float(raw["delay_step_sec"])
    delays = np.arange(
        float(raw["delay_min_sec"]),
        float(raw["delay_max_sec"]) + 0.5 * delay_step,
        delay_step,
    )
    if np.any(~np.isclose(delays / dt, np.round(delays / dt))):
        raise ValueError("delay 후보는 physics_dt의 정수배여야 합니다.")
    refine_count = int(raw["delay_refine_count"])
    if not 1 <= refine_count <= len(delays):
        raise ValueError("delay_refine_count는 delay 후보 수 이하여야 합니다.")
    return EquivalentFitConfig(
        source=source,
        replay=EquivalentReplayConfig(
            model_xml=Path(raw["model_xml"]),
            physics_dt_sec=dt,
            torque_limit_nm=float(raw["torque_limit_nm"]),
        ),
        target_joints=(targets[0], targets[1]),
        pose_id=str(raw["pose_id"]),
        conditions=(conditions[0], conditions[1]),
        kp_per_register_bounds=_pair(raw, "kp_per_register_bounds"),
        initial_kp_per_register=initial_scale,
        fixed_kd_eff=fixed_kd,
        backlash_bounds_rad=_pair(raw, "backlash_total_rad_bounds"),
        coulomb_bounds_nm=_pair(raw, "coulomb_friction_nm_bounds"),
        delay_values_sec=delays,
        delay_refine_count=refine_count,
        transient_window_sec=float(raw["transient_window_sec"]),
        plateau_tail_sec=float(raw["plateau_tail_sec"]),
        huber_delta_normalized=float(raw["huber_delta_normalized"]),
        velocity_loss_weight=float(raw["velocity_loss_weight"]),
        plateau_loss_weight=float(raw["plateau_loss_weight"]),
        stage1_max_evaluations=int(raw["stage1_max_evaluations"]),
        stage2_max_evaluations=int(raw["stage2_max_evaluations"]),
        final_max_evaluations=int(raw["final_max_evaluations"]),
        final_starts=int(raw["final_starts"]),
    )


def _edge_segments(run: RunData, config: EquivalentFitConfig):
    command = run.q_cmd_rad[:, run.target_index]
    edges = np.flatnonzero(np.abs(np.diff(command)) > 1e-12) + 1
    transient_count = max(1, round(config.transient_window_sec * run.command_rate_hz))
    tail_count = max(1, round(config.plateau_tail_sec * run.command_rate_hz))
    result = []
    for number, edge in enumerate(edges):
        next_edge = int(edges[number + 1]) if number + 1 < len(edges) else len(command)
        transient_end = min(next_edge, int(edge) + transient_count)
        tail_start = max(int(edge), next_edge - tail_count)
        if transient_end > edge and next_edge > tail_start:
            result.append((int(edge), transient_end, tail_start, next_edge))
    if not result:
        raise ValueError(f"step edge가 없습니다: {run.run_dir}")
    return tuple(result)


def equivalent_loss_parts(
    run: RunData,
    result: EquivalentReplayResult,
    config: EquivalentFitConfig,
) -> dict[str, float]:
    """과도응답과 절대 plateau 오차를 edge별 동일 비중으로 계산한다."""
    index = run.target_index
    command = run.q_cmd_rad[:, index]
    dynamic_parts: list[float] = []
    velocity_parts: list[float] = []
    plateau_parts: list[float] = []
    dq_scale = max(
        float(np.nanpercentile(np.abs(run.dq_real_rad_s[run.state_mask, index]), 95)),
        0.05,
    )
    for edge, end, tail_start, tail_end in _edge_segments(run, config):
        transient_mask = run.state_mask[edge:end]
        tail_mask = run.state_mask[tail_start:tail_end]
        if not transient_mask.any() or not tail_mask.any():
            continue
        scale = max(abs(float(command[edge] - command[edge - 1])), 0.01)
        real_plateau = float(np.nanmedian(run.q_real_rad[tail_start:tail_end, index][tail_mask]))
        sim_plateau = float(np.median(result.q_rad[tail_start:tail_end, index][tail_mask]))
        real_q = run.q_real_rad[edge:end, index][transient_mask] - real_plateau
        sim_q = result.q_rad[edge:end, index][transient_mask] - sim_plateau
        dynamic_parts.append(
            float(np.mean(_huber((sim_q - real_q) / scale, config.huber_delta_normalized)))
        )
        real_dq = run.dq_real_rad_s[edge:end, index][transient_mask]
        sim_dq = result.dq_rad_s[edge:end, index][transient_mask]
        velocity_parts.append(
            float(np.mean(_huber((sim_dq - real_dq) / dq_scale, config.huber_delta_normalized)))
        )
        plateau_parts.append(
            float(_huber(np.asarray([(sim_plateau - real_plateau) / scale]), config.huber_delta_normalized)[0])
        )
    if not dynamic_parts:
        raise ValueError(f"유효한 edge가 없습니다: {run.run_dir}")
    return {
        "dynamic": float(np.mean(dynamic_parts)),
        "velocity": float(np.mean(velocity_parts)),
        "plateau": float(np.mean(plateau_parts)),
    }


def _combined(parts: dict[str, float], config: EquivalentFitConfig) -> float:
    return (
        parts["dynamic"]
        + config.velocity_loss_weight * parts["velocity"]
        + config.plateau_loss_weight * parts["plateau"]
    )


def _tied_pd(
    kp_per_register: float,
    config: EquivalentFitConfig,
) -> dict[str, ConditionPd]:
    return {
        condition.name: ConditionPd(
            kp=kp_per_register * condition.register_p,
            kd=config.fixed_kd_eff,
        )
        for condition in config.conditions
    }


def _mean_loss(
    simulator: FixedBaseEquivalentPdReplay,
    runs_by_condition: dict[str, tuple[RunData, ...]],
    config: EquivalentFitConfig,
    delay: float,
    pd: dict[str, ConditionPd],
    backlash: float,
    coulomb: float,
    *,
    include_plateau: bool,
) -> float:
    losses = []
    condition_map = {condition.name: condition for condition in config.conditions}
    for condition_name, runs in runs_by_condition.items():
        condition = condition_map[condition_name]
        parameters = EquivalentParameters(
            pd[condition.name].kp,
            pd[condition.name].kd,
            backlash,
            coulomb,
        )
        for run in runs:
            parts = equivalent_loss_parts(
                run,
                simulator.run(run, delay_sec=delay, parameters=parameters),
                config,
            )
            if not include_plateau:
                parts["plateau"] = 0.0
            losses.append(_combined(parts, config))
    return float(np.mean(losses))


def load_all_joint_validation_runs(
    campaign_root: str | Path,
    config: EquivalentFitConfig,
    expected_register_p: int,
) -> tuple[RunData, ...]:
    """12개 관절의 repeat 3 step을 순수 validation용으로 읽는다."""
    root = Path(campaign_root).expanduser().resolve()
    completed = json.loads((root / "campaign_status.json").read_text())["completed"]
    runs: list[RunData] = []
    for joint in MUJOCO_DOF_ORDER:
        key = f"multi_amplitude_step/{joint}/3/validation"
        if key not in completed:
            raise ValueError(f"campaign status에 validation run이 없습니다: {key}")
        run = load_run(root / "runs" / completed[key])
        actual_p = {
            int(values["position_p_gain"])
            for values in run.metadata.get("actuator_settings", {}).values()
        }
        if (
            run.target_joint != joint
            or run.repeat_index != 3
            or run.split_role != "validation"
            or run.metadata.get("experiment_type") != "compact_step"
            or run.metadata.get("pose_id") != config.pose_id
            or actual_p != {expected_register_p}
        ):
            raise ValueError(f"12관절 validation metadata 불일치: {run.run_dir}")
        runs.append(run)
    return tuple(runs)


def _fit_stage1(
    simulator: FixedBaseEquivalentPdReplay,
    fit_runs: dict[str, tuple[RunData, ...]],
    config: EquivalentFitConfig,
):
    initial_pd = _tied_pd(
        config.initial_kp_per_register,
        config,
    )
    screening = []
    for delay in config.delay_values_sec:
        loss = _mean_loss(
            simulator, fit_runs, config, float(delay), initial_pd,
            0.0, 0.0, include_plateau=False,
        )
        screening.append((float(delay), loss))
        print(f"stage1 screen delay={delay*1000:4.1f} ms loss={loss:.8f}")
    refine_delays = tuple(
        delay for delay, _ in sorted(screening, key=lambda item: item[1])[
            : config.delay_refine_count
        ]
    )
    print(
        "stage1 refine delays: "
        + ", ".join(f"{delay*1000:.1f} ms" for delay in refine_delays)
    )
    refined = []
    for delay in refine_delays:
        def objective(x):
            pd = _tied_pd(float(x[0]), config)
            return _mean_loss(
                simulator, fit_runs, config, float(delay), pd,
                0.0, 0.0, include_plateau=False,
            )
        result = minimize(
            objective,
            [config.initial_kp_per_register],
            method="Powell",
            bounds=(config.kp_per_register_bounds,),
            options={"maxfev": config.stage1_max_evaluations, "xtol": 1e-5, "ftol": 1e-6},
        )
        by_condition = _tied_pd(float(result.x[0]), config)
        loss = float(result.fun)
        refined.append((float(delay), by_condition, loss))
        detail = "  ".join(
            f"{name}:Kp={value.kp:.4f},Kd={value.kd:.4f}"
            for name, value in by_condition.items()
        )
        print(
            f"stage1 refine delay={delay*1000:4.1f} ms loss={loss:.8f}  "
            f"Kp/register={result.x[0]:.7f}, fixed Kd={config.fixed_kd_eff:.4f}  {detail}"
        )
    return min(refined, key=lambda item: item[2]), tuple(screening), tuple(refined)


def _fit_static(
    simulator, fit_runs, config, delay, pd
):
    def objective(x):
        return _mean_loss(
            simulator, fit_runs, config, delay, pd,
            float(x[0]), float(x[1]), include_plateau=True,
        )
    result = minimize(
        objective,
        [0.012, 0.08],
        method="Powell",
        bounds=(config.backlash_bounds_rad, config.coulomb_bounds_nm),
        options={"maxfev": config.stage2_max_evaluations, "xtol": 1e-4, "ftol": 1e-6},
    )
    print(
        f"stage2 backlash={result.x[0]:.6f} rad, Coulomb={result.x[1]:.6f} Nm, "
        f"loss={result.fun:.8f}"
    )
    return float(result.x[0]), float(result.x[1])


def _candidate_from_vector(delay, x, loss, success, evaluations, config):
    pd = _tied_pd(float(x[0]), config)
    return EquivalentCandidate(
        delay_sec=delay,
        by_condition=pd,
        backlash_total_rad=float(x[1]),
        coulomb_friction_nm=float(x[2]),
        loss=float(loss), success=bool(success), evaluations=int(evaluations),
        kp_per_register=float(x[0]),
        shared_kd=config.fixed_kd_eff,
    )


def _fit_final(simulator, fit_runs, config, delay, pd, backlash, coulomb):
    scales = [
        pd[condition.name].kp / condition.register_p
        for condition in config.conditions
    ]
    base = np.asarray([
        float(np.mean(scales)),
        backlash, coulomb,
    ])
    starts = [base]
    if config.final_starts > 1:
        starts.append(base * np.asarray([1.10, 1.15, 0.85]))
    bounds = (
        config.kp_per_register_bounds,
        config.backlash_bounds_rad,
        config.coulomb_bounds_nm,
    )
    candidates = []
    for number, start in enumerate(starts[:config.final_starts], 1):
        start = np.asarray([np.clip(value, bound[0], bound[1]) for value, bound in zip(start, bounds)])
        def objective(x):
            trial_pd = _tied_pd(float(x[0]), config)
            return _mean_loss(
                simulator, fit_runs, config, delay, trial_pd,
                float(x[1]), float(x[2]), include_plateau=True,
            )
        result = minimize(
            objective, start, method="Powell", bounds=bounds,
            options={"maxfev": config.final_max_evaluations, "xtol": 1e-4, "ftol": 1e-6},
        )
        candidate = _candidate_from_vector(delay, result.x, result.fun, result.success, result.nfev, config)
        candidates.append(candidate)
        print(
            f"stage3 start {number}: loss={candidate.loss:.8f}, "
            f"Kp/register={candidate.kp_per_register:.7f}, "
            f"Kd={candidate.shared_kd:.4f}, "
            f"backlash={candidate.backlash_total_rad:.6f}, "
            f"Coulomb={candidate.coulomb_friction_nm:.6f}, "
            f"success={candidate.success}, n={candidate.evaluations}"
        )
    return min(candidates, key=lambda item: item.loss), tuple(candidates)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_plot(path, run, result, condition, best):
    index = run.target_index
    time = np.arange(len(run.q_cmd_rad)) / run.command_rate_hz
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(time, run.q_cmd_rad[:, index], "k--", lw=1.0, label="q_cmd")
    axis.plot(time[run.state_mask], run.q_real_rad[run.state_mask, index], lw=1.2, label="real")
    axis.plot(time, result.q_rad[:, index], color="tab:red", lw=1.0, label="equivalent PD")
    axis.set(xlabel="time [s]", ylabel="joint angle [rad]", title=f"{condition} delay={best.delay_sec*1000:.1f} ms")
    axis.grid(alpha=0.25); axis.legend(); figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def _metric_row(run, replay, config):
    parts = equivalent_loss_parts(run, replay, config)
    mask = run.state_mask
    index = run.target_index
    error = replay.q_rad[mask, index] - run.q_real_rad[mask, index]
    return {
        "run_dir": str(run.run_dir),
        "joint": run.target_joint,
        "repeat_index": run.repeat_index,
        "split_role": run.split_role,
        **parts,
        "combined_loss": _combined(parts, config),
        "full_mae_rad": float(np.mean(np.abs(error))),
        "full_rmse_rad": float(np.sqrt(np.mean(error**2))),
    }


def identify_equivalent_pd(campaigns, config, output_root):
    all_runs = {}; fit_runs = {}; validation_runs = {}
    all_joint_validation = {}
    for condition in config.conditions:
        runs = load_collection_campaign_runs(campaigns[condition.name], config, condition.register_p)
        fit, validation = split_fit_validation(runs)
        all_runs[condition.name] = runs; fit_runs[condition.name] = fit; validation_runs[condition.name] = validation
        all_joint_validation[condition.name] = load_all_joint_validation_runs(
            campaigns[condition.name], config, condition.register_p
        )
    simulator = FixedBaseEquivalentPdReplay(config.replay)
    (delay, pd, _), screening, refined = _fit_stage1(simulator, fit_runs, config)
    backlash, coulomb = _fit_static(simulator, fit_runs, config, delay, pd)
    best, final_candidates = _fit_final(simulator, fit_runs, config, delay, pd, backlash, coulomb)

    output = Path(output_root).expanduser().resolve() / f"{datetime.now():%Y%m%d_%H%M%S}_equivalent_pd"
    output.mkdir(parents=True, exist_ok=False)
    metrics: dict[str, Any] = {}
    for condition in config.conditions:
        parameters = EquivalentParameters(
            best.by_condition[condition.name].kp, best.by_condition[condition.name].kd,
            best.backlash_total_rad, best.coulomb_friction_nm,
        )
        rows = []
        for run in all_runs[condition.name]:
            replay = simulator.run(run, delay_sec=best.delay_sec, parameters=parameters)
            row = _metric_row(run, replay, config)
            rows.append(row)
            _save_plot(output / f"{condition.name}_{run.target_joint}_repeat{run.repeat_index}_{run.split_role}.png", run, replay, condition.name, best)
        metrics[condition.name] = rows
        for role in ("fit", "validation"):
            selected = [row for row in rows if row["split_role"] == role]
            metrics[f"{condition.name}_{role}_mean"] = {
                key: float(np.mean([row[key] for row in selected]))
                for key in ("dynamic", "velocity", "plateau", "combined_loss", "full_mae_rad", "full_rmse_rad")
            }

        heldout_rows = []
        for run in all_joint_validation[condition.name]:
            replay = simulator.run(run, delay_sec=best.delay_sec, parameters=parameters)
            heldout_rows.append(_metric_row(run, replay, config))
            _save_plot(
                output / f"all_joint_validation_{condition.name}_{run.target_joint}.png",
                run, replay, condition.name, best,
            )
        metrics[f"{condition.name}_all_joint_validation"] = heldout_rows
        metrics[f"{condition.name}_all_joint_validation_mean"] = {
            key: float(np.mean([row[key] for row in heldout_rows]))
            for key in ("dynamic", "velocity", "plateau", "combined_loss", "full_mae_rad", "full_rmse_rad")
        }

    params = {
        "model": "register_tied_equivalent_PD_deadband_Coulomb",
        "shared_delay_sec": best.delay_sec,
        "shared_delay_ms": best.delay_sec * 1000.0,
        "shared_backlash_total_rad": best.backlash_total_rad,
        "shared_backlash_total_deg": float(np.degrees(best.backlash_total_rad)),
        "shared_coulomb_friction_nm": best.coulomb_friction_nm,
        "kp_eff_per_position_p_register": best.kp_per_register,
        "fixed_kd_eff": best.shared_kd,
        "conditions": {
            c.name: {"register_p": c.register_p, "kp_eff": best.by_condition[c.name].kp, "kd_eff": best.by_condition[c.name].kd}
            for c in config.conditions
        },
    }
    (output / "params_equivalent_pd.yaml").write_text(yaml.safe_dump(params, sort_keys=False, allow_unicode=True))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "optimization.json").write_text(json.dumps({
        "stage1_screen": [{"delay_sec": d, "loss": loss} for d, loss in screening],
        "stage1_refined": [{"delay_sec": d, "loss": loss, "pd": {n: {"kp": p.kp, "kd": p.kd} for n, p in values.items()}} for d, values, loss in refined],
        "stage3": [{
            "loss": c.loss, "success": c.success, "evaluations": c.evaluations,
            "kp_per_register": c.kp_per_register, "fixed_kd": c.shared_kd,
            "backlash_total_rad": c.backlash_total_rad,
            "coulomb_friction_nm": c.coulomb_friction_nm,
        } for c in final_candidates],
    }, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "fit_config": str(config.source), "fit_config_sha256": _sha256(config.source),
        "model_xml": str(config.replay.model_xml.expanduser().resolve()),
        "model_xml_sha256": _sha256(config.replay.model_xml.expanduser().resolve()),
        "campaigns": {name: str(path.expanduser().resolve()) for name, path in campaigns.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# Jandi equivalent PD actuator identification", "",
        f"- shared delay: {best.delay_sec*1000:.3f} ms",
        f"- equivalent backlash total width: {best.backlash_total_rad:.6f} rad ({np.degrees(best.backlash_total_rad):.3f} deg)",
        f"- MuJoCo Coulomb frictionloss: {best.coulomb_friction_nm:.6f} Nm",
        f"- Kp_eff / Position-P register: {best.kp_per_register:.9f}",
        f"- fixed Kd_eff: {best.shared_kd:.6f}",
    ]
    for c in config.conditions:
        p = best.by_condition[c.name]
        report.append(f"- {c.name}: Kp_eff={p.kp:.6f}, Kd_eff={p.kd:.6f}")
    report += ["", "## 12-joint held-out validation", "", "| condition | joint | MAE deg | RMSE deg | combined loss |", "|---|---:|---:|---:|---:|"]
    for condition in config.conditions:
        for row in metrics[f"{condition.name}_all_joint_validation"]:
            report.append(
                f"| {condition.name} | {row['joint']} | "
                f"{np.degrees(row['full_mae_rad']):.4f} | "
                f"{np.degrees(row['full_rmse_rad']):.4f} | "
                f"{row['combined_loss']:.8f} |"
            )
    report += ["", "## Model contract", "", "- Kp는 Position-P register에 정비례하도록 묶었습니다.", "- Kd=0.60은 고정했으며 최적화하지 않았습니다.", "- RL6·LL6 repeat 1·2만 fit에 사용했습니다.", "- 12개 관절 repeat 3은 재피팅 없는 held-out validation입니다.", "- 각 조건의 PD는 12개 관절 전체에 적용했습니다.", "- backlash 값은 상태를 가진 기어 치합 모델이 아니라 위치 오차의 등가 deadband 전체 폭입니다.", "- 정상상태 plateau 절대오차를 loss에 포함했습니다.", "- viscous friction은 Kd와 식별 불가능하므로 별도 피팅하지 않았습니다."]
    (output / "report.md").write_text("\n".join(report) + "\n")
    return output
