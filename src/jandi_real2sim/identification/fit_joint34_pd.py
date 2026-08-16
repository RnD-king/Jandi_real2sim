from __future__ import annotations

import hashlib
import json
import time
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
from jandi_real2sim.identification.dataset import RunData, load_run
from jandi_real2sim.identification.fit_m0 import _huber
from jandi_real2sim.identification.replay_equivalent_pd import (
    EquivalentReplayConfig,
    EquivalentReplayResult,
    FixedBaseStatefulJointwisePdReplay,
    JointwiseEquivalentParameters,
)


@dataclass(frozen=True)
class GainCondition:
    name: str
    register_p: int


@dataclass(frozen=True)
class Joint34FitConfig:
    source: Path
    replay: EquivalentReplayConfig
    pose_id: str
    group_names: tuple[str, str]
    groups: dict[str, tuple[str, str]]
    fit_condition: GainCondition
    validation_conditions: tuple[GainCondition, ...]
    reference_register_p: int
    fixed_base_kp_at_reference: float
    fixed_base_kd: float
    fixed_delay_sec: float
    fixed_backlash_total_rad: float
    fixed_coulomb_friction_nm: float
    position_quantization_rad: float
    initial_kp: dict[str, float]
    kp_bounds: dict[str, tuple[float, float]]
    initial_kd: dict[str, float]
    kd_bounds: dict[str, tuple[float, float]]
    loss: dict[str, float]
    optimizer: dict[str, int]

    @property
    def target_joints(self) -> tuple[str, ...]:
        return tuple(
            joint
            for group_name in self.group_names
            for joint in self.groups[group_name]
        )


@dataclass(frozen=True)
class CampaignRuns:
    condition: GainCondition
    root: Path
    fit: dict[str, tuple[RunData, ...]]
    validation: dict[str, tuple[RunData, ...]]


@dataclass(frozen=True)
class Joint34Candidate:
    kp_at_reference: dict[str, float]
    kd: dict[str, float]
    loss: float
    success: bool
    evaluations: int


def _positive_pair(raw: Any, label: str) -> tuple[float, float]:
    values = tuple(map(float, raw))
    if len(values) != 2 or values[0] < 0.0 or values[0] >= values[1]:
        raise ValueError(f"{label} 범위가 잘못됐습니다: {values}")
    return values


def load_joint34_fit_config(path: str | Path) -> Joint34FitConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text())
    group_names = tuple(map(str, raw["joint_groups"]))
    if len(group_names) != 2:
        raise ValueError("joint_groups는 정확히 두 그룹이어야 합니다.")
    groups: dict[str, tuple[str, str]] = {}
    used: set[str] = set()
    for group_name in group_names:
        joints = tuple(map(str, raw["joint_groups"][group_name]))
        if (
            len(joints) != 2
            or any(joint not in MUJOCO_DOF_ORDER for joint in joints)
            or used.intersection(joints)
        ):
            raise ValueError(
                f"{group_name}은 중복 없는 유효한 좌우 두 관절이어야 합니다."
            )
        groups[group_name] = (joints[0], joints[1])
        used.update(joints)
    fit_raw = raw["fit_condition"]
    fit_condition = GainCondition(
        str(fit_raw["name"]),
        int(fit_raw["register_p"]),
    )
    validation_conditions = tuple(
        GainCondition(str(item["name"]), int(item["register_p"]))
        for item in raw.get("validation_conditions", ())
    )
    if fit_condition.name in {item.name for item in validation_conditions}:
        raise ValueError("fit/validation condition 이름이 중복됩니다.")
    initial_kp = {
        name: float(raw["initial_kp_at_reference"][name])
        for name in group_names
    }
    initial_kd = {
        name: float(raw["initial_kd"][name]) for name in group_names
    }
    kp_bounds = {
        name: _positive_pair(
            raw["kp_at_reference_bounds"][name],
            f"kp_at_reference_bounds.{name}",
        )
        for name in group_names
    }
    kd_bounds = {
        name: _positive_pair(raw["kd_bounds"][name], f"kd_bounds.{name}")
        for name in group_names
    }
    for name in group_names:
        if not kp_bounds[name][0] <= initial_kp[name] <= kp_bounds[name][1]:
            raise ValueError(f"{name} initial Kp가 bounds 밖입니다.")
        if not kd_bounds[name][0] <= initial_kd[name] <= kd_bounds[name][1]:
            raise ValueError(f"{name} initial Kd가 bounds 밖입니다.")
    loss = {str(k): float(v) for k, v in raw["loss"].items()}
    optimizer = {str(k): int(v) for k, v in raw["optimizer"].items()}
    reference_p = int(raw["reference_register_p"])
    if reference_p <= 0 or fit_condition.register_p != reference_p:
        raise ValueError(
            "현재 식별기는 fit_condition과 reference_register_p가 같아야 합니다."
        )
    return Joint34FitConfig(
        source=source,
        replay=EquivalentReplayConfig(
            model_xml=Path(raw["model_xml"]),
            physics_dt_sec=float(raw["physics_dt_sec"]),
            torque_limit_nm=float(raw["torque_limit_nm"]),
        ),
        pose_id=str(raw["pose_id"]),
        group_names=(group_names[0], group_names[1]),
        groups=groups,
        fit_condition=fit_condition,
        validation_conditions=validation_conditions,
        reference_register_p=reference_p,
        fixed_base_kp_at_reference=float(raw["fixed_base_kp_at_reference"]),
        fixed_base_kd=float(raw["fixed_base_kd"]),
        fixed_delay_sec=float(raw["fixed_delay_sec"]),
        fixed_backlash_total_rad=float(raw["fixed_backlash_total_rad"]),
        fixed_coulomb_friction_nm=float(raw["fixed_coulomb_friction_nm"]),
        position_quantization_rad=float(raw["position_quantization_rad"]),
        initial_kp=initial_kp,
        kp_bounds=kp_bounds,
        initial_kd=initial_kd,
        kd_bounds=kd_bounds,
        loss=loss,
        optimizer=optimizer,
    )


def _validate_register(run: RunData, expected_register_p: int) -> None:
    actual = {
        int(values["position_p_gain"])
        for values in run.metadata.get("actuator_settings", {}).values()
    }
    if actual != {expected_register_p}:
        raise ValueError(
            f"{run.run_dir}: Position P={actual}, expected={expected_register_p}"
        )


def _load_campaign(
    root: str | Path,
    config: Joint34FitConfig,
    condition: GainCondition,
) -> CampaignRuns:
    root = Path(root).expanduser().resolve()
    completed = json.loads((root / "campaign_status.json").read_text())[
        "completed"
    ]
    fit: dict[str, list[RunData]] = {
        name: []
        for name in ("compact_step", "triangle", "multisine", "static_hold")
    }
    validation = {name: [] for name in fit}
    experiment_specs = (
        ("compact_step", "multi_amplitude_step", "repeat"),
        ("triangle", "slow_triangle", "repeat"),
        ("multisine", "policy_band_multisine", "seed"),
    )
    for experiment_type, campaign_name, _ in experiment_specs:
        for joint in config.target_joints:
            for number in (1, 2, 3):
                role = (
                    "diagnostic"
                    if experiment_type == "triangle"
                    else ("fit" if number in (1, 2) else "validation")
                )
                key = f"{campaign_name}/{joint}/{number}/{role}"
                if key not in completed:
                    raise ValueError(f"campaign status에 run이 없습니다: {key}")
                run = load_run(root / "runs" / completed[key])
                if (
                    run.metadata.get("experiment_type") != experiment_type
                    or run.metadata.get("pose_id") != config.pose_id
                    or run.target_joint != joint
                    or run.repeat_index != number
                ):
                    raise ValueError(f"run metadata 불일치: {run.run_dir}")
                _validate_register(run, condition.register_p)
                (fit if number in (1, 2) else validation)[
                    experiment_type
                ].append(run)

    hold_target = config.target_joints[0]
    for repeat in (1, 2, 3):
        key = f"walking_pose_hold/all_joints/{repeat}/baseline"
        if key not in completed:
            raise ValueError(f"campaign status에 static hold가 없습니다: {key}")
        run = load_run(
            root / "runs" / completed[key],
            target_joint_override=hold_target,
        )
        if (
            run.metadata.get("experiment_type") != "static_hold"
            or run.metadata.get("pose_id") != config.pose_id
            or run.repeat_index != repeat
        ):
            raise ValueError(f"static hold metadata 불일치: {run.run_dir}")
        _validate_register(run, condition.register_p)
        (fit if repeat in (1, 2) else validation)["static_hold"].append(
            run
        )
    return CampaignRuns(
        condition=condition,
        root=root,
        fit={name: tuple(runs) for name, runs in fit.items()},
        validation={name: tuple(runs) for name, runs in validation.items()},
    )


def _parameters(
    config: Joint34FitConfig,
    condition: GainCondition,
    kp_at_reference: dict[str, float],
    kd: dict[str, float],
) -> JointwiseEquivalentParameters:
    ratio = condition.register_p / config.reference_register_p
    kp = np.full(12, config.fixed_base_kp_at_reference * ratio)
    damping = np.full(12, config.fixed_base_kd)
    for group_name in config.group_names:
        for joint in config.groups[group_name]:
            index = MUJOCO_DOF_ORDER.index(joint)
            kp[index] = kp_at_reference[group_name] * ratio
            damping[index] = kd[group_name]
    return JointwiseEquivalentParameters(
        kp_eff=kp,
        kd_eff=damping,
        backlash_total_rad=config.fixed_backlash_total_rad,
        coulomb_friction_nm=config.fixed_coulomb_friction_nm,
        position_quantization_rad=config.position_quantization_rad,
    )


def _mean_huber(values: np.ndarray, scale: float, delta: float) -> float:
    if scale <= 0.0:
        raise ValueError("loss scale은 양수여야 합니다.")
    return float(np.mean(_huber(values / scale, delta)))


def _edge_segments(
    run: RunData,
    transient_sec: float,
    tail_sec: float,
) -> tuple[tuple[int, int, int, int], ...]:
    command = run.q_cmd_rad[:, run.target_index]
    edges = np.flatnonzero(np.abs(np.diff(command)) > 1e-12) + 1
    transient_count = max(1, round(transient_sec * run.command_rate_hz))
    tail_count = max(1, round(tail_sec * run.command_rate_hz))
    segments = []
    for number, edge in enumerate(edges):
        next_edge = int(edges[number + 1]) if number + 1 < len(edges) else len(command)
        transient_end = min(next_edge, int(edge) + transient_count)
        tail_start = max(int(edge), next_edge - tail_count)
        if transient_end > edge and next_edge > tail_start:
            segments.append((int(edge), transient_end, tail_start, next_edge))
    if not segments:
        raise ValueError(f"step edge가 없습니다: {run.run_dir}")
    return tuple(segments)


def _triangle_loss(
    run: RunData,
    replay: EquivalentReplayResult,
    config: Joint34FitConfig,
) -> float:
    index = run.target_index
    mask = run.state_mask
    scale = max(float(run.metadata["amplitude_rad"]), 0.01)
    return _mean_huber(
        replay.q_rad[mask, index] - run.q_real_rad[mask, index],
        scale,
        config.loss["huber_delta_normalized"],
    )


def _hold_loss(
    run: RunData,
    replay: EquivalentReplayResult,
    config: Joint34FitConfig,
) -> float:
    start = round(
        len(run.state_mask) * (1.0 - config.loss["static_tail_fraction"])
    )
    mask = run.state_mask.copy()
    mask[:start] = False
    indices = np.asarray(
        [MUJOCO_DOF_ORDER.index(joint) for joint in config.target_joints]
    )
    real = run.q_real_rad[mask][:, indices]
    sim = replay.q_rad[mask][:, indices]
    return _mean_huber(
        sim - real,
        config.loss["static_position_scale_rad"],
        config.loss["huber_delta_normalized"],
    )


def _step_plateau_loss(
    run: RunData,
    replay: EquivalentReplayResult,
    config: Joint34FitConfig,
) -> float:
    index = run.target_index
    command = run.q_cmd_rad[:, index]
    losses = []
    for edge, _, tail_start, tail_end in _edge_segments(
        run,
        config.loss["step_transient_window_sec"],
        config.loss["step_plateau_tail_sec"],
    ):
        mask = run.state_mask[tail_start:tail_end]
        if not mask.any():
            continue
        scale = max(abs(float(command[edge] - command[edge - 1])), 0.01)
        real = run.q_real_rad[tail_start:tail_end, index][mask]
        sim = replay.q_rad[tail_start:tail_end, index][mask]
        losses.append(
            _mean_huber(
                sim - real,
                scale,
                config.loss["huber_delta_normalized"],
            )
        )
    return float(np.mean(losses))


def _step_dynamic_loss(
    run: RunData,
    replay: EquivalentReplayResult,
    config: Joint34FitConfig,
) -> float:
    index = run.target_index
    command = run.q_cmd_rad[:, index]
    q_parts = []
    dq_parts = []
    dq_scale = max(
        float(np.nanpercentile(np.abs(run.dq_real_rad_s[run.state_mask, index]), 95)),
        0.05,
    )
    for edge, end, tail_start, tail_end in _edge_segments(
        run,
        config.loss["step_transient_window_sec"],
        config.loss["step_plateau_tail_sec"],
    ):
        transient_mask = run.state_mask[edge:end]
        tail_mask = run.state_mask[tail_start:tail_end]
        if not transient_mask.any() or not tail_mask.any():
            continue
        scale = max(abs(float(command[edge] - command[edge - 1])), 0.01)
        real_plateau = float(
            np.nanmedian(
                run.q_real_rad[tail_start:tail_end, index][tail_mask]
            )
        )
        sim_plateau = float(
            np.median(replay.q_rad[tail_start:tail_end, index][tail_mask])
        )
        real_q = run.q_real_rad[edge:end, index][transient_mask] - real_plateau
        sim_q = replay.q_rad[edge:end, index][transient_mask] - sim_plateau
        q_parts.append(
            _mean_huber(
                sim_q - real_q,
                scale,
                config.loss["huber_delta_normalized"],
            )
        )
        real_dq = run.dq_real_rad_s[edge:end, index][transient_mask]
        sim_dq = replay.dq_rad_s[edge:end, index][transient_mask]
        dq_parts.append(
            _mean_huber(
                sim_dq - real_dq,
                dq_scale,
                config.loss["huber_delta_normalized"],
            )
        )
    return float(
        np.mean(q_parts)
        + config.loss["step_velocity_weight"] * np.mean(dq_parts)
    )


def _multisine_loss(
    run: RunData,
    replay: EquivalentReplayResult,
    config: Joint34FitConfig,
) -> float:
    index = run.target_index
    mask = run.state_mask
    scale = max(float(run.metadata["amplitude_rad"]), 0.01)
    real_q = run.q_real_rad[mask, index]
    sim_q = replay.q_rad[mask, index]
    real_q = real_q - np.nanmedian(real_q)
    sim_q = sim_q - np.median(sim_q)
    q_loss = _mean_huber(
        sim_q - real_q,
        scale,
        config.loss["huber_delta_normalized"],
    )
    real_dq = run.dq_real_rad_s[mask, index]
    sim_dq = replay.dq_rad_s[mask, index]
    dq_scale = max(float(np.nanpercentile(np.abs(real_dq), 95)), 0.05)
    dq_loss = _mean_huber(
        sim_dq - real_dq,
        dq_scale,
        config.loss["huber_delta_normalized"],
    )
    return q_loss + config.loss["multisine_velocity_weight"] * dq_loss


def _weighted_mean(items: tuple[tuple[float, float], ...]) -> float:
    active = [(value, weight) for value, weight in items if weight > 0.0]
    return float(
        sum(value * weight for value, weight in active)
        / sum(weight for _, weight in active)
    )


def _evaluate_losses(
    simulator: FixedBaseStatefulJointwisePdReplay,
    runs: dict[str, tuple[RunData, ...]],
    config: Joint34FitConfig,
    condition: GainCondition,
    kp: dict[str, float],
    kd: dict[str, float],
    trajectories: tuple[str, ...],
) -> dict[str, float]:
    parameters = _parameters(config, condition, kp, kd)
    parts: dict[str, float] = {}
    for trajectory in trajectories:
        values = []
        for run in runs[trajectory]:
            replay = simulator.run(
                run,
                delay_sec=config.fixed_delay_sec,
                parameters=parameters,
            )
            if trajectory == "triangle":
                value = _triangle_loss(run, replay, config)
            elif trajectory == "static_hold":
                value = _hold_loss(run, replay, config)
            elif trajectory == "compact_step":
                value = _step_dynamic_loss(run, replay, config)
            elif trajectory == "multisine":
                value = _multisine_loss(run, replay, config)
            else:
                raise ValueError(f"알 수 없는 trajectory: {trajectory}")
            values.append(value)
        parts[trajectory] = float(np.mean(values))
    return parts


def _evaluate_quasistatic(
    simulator: FixedBaseStatefulJointwisePdReplay,
    runs: dict[str, tuple[RunData, ...]],
    config: Joint34FitConfig,
    condition: GainCondition,
    kp: dict[str, float],
    kd: dict[str, float],
) -> tuple[float, dict[str, float]]:
    parameters = _parameters(config, condition, kp, kd)
    parts: dict[str, float] = {}
    for trajectory, loss_fn in (
        ("triangle", _triangle_loss),
        ("static_hold", _hold_loss),
    ):
        parts[trajectory] = float(
            np.mean(
                [
                    loss_fn(
                        run,
                        simulator.run(
                            run,
                            delay_sec=config.fixed_delay_sec,
                            parameters=parameters,
                        ),
                        config,
                    )
                    for run in runs[trajectory]
                ]
            )
        )
    plateau = []
    for run in runs["compact_step"]:
        replay = simulator.run(
            run,
            delay_sec=config.fixed_delay_sec,
            parameters=parameters,
        )
        plateau.append(_step_plateau_loss(run, replay, config))
    parts["step_plateau"] = float(np.mean(plateau))
    total = _weighted_mean(
        (
            (parts["triangle"], config.loss["triangle_weight"]),
            (parts["static_hold"], config.loss["static_hold_weight"]),
            (parts["step_plateau"], config.loss["step_plateau_weight"]),
        )
    )
    return total, parts


def _evaluate_dynamic(
    simulator: FixedBaseStatefulJointwisePdReplay,
    runs: dict[str, tuple[RunData, ...]],
    config: Joint34FitConfig,
    condition: GainCondition,
    kp: dict[str, float],
    kd: dict[str, float],
) -> tuple[float, dict[str, float]]:
    parts = _evaluate_losses(
        simulator,
        runs,
        config,
        condition,
        kp,
        kd,
        ("compact_step", "multisine"),
    )
    total = _weighted_mean(
        (
            (parts["compact_step"], config.loss["step_dynamic_weight"]),
            (parts["multisine"], config.loss["multisine_dynamic_weight"]),
        )
    )
    return total, parts


def _dict_from_vector(
    names: tuple[str, str],
    values: np.ndarray,
) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values)}


def _optimize_stage(
    name: str,
    initial: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    max_evaluations: int,
    objective_fn,
):
    cache: dict[tuple[float, ...], float] = {}
    evaluations = 0
    started = time.monotonic()

    def objective(values):
        nonlocal evaluations
        key = tuple(float(value) for value in values)
        if key in cache:
            return cache[key]
        evaluations += 1
        loss = float(objective_fn(np.asarray(values, dtype=np.float64)))
        cache[key] = loss
        if evaluations == 1 or evaluations % 10 == 0:
            elapsed = time.monotonic() - started
            detail = ", ".join(f"{value:.5f}" for value in values)
            print(
                f"[{name}] eval={evaluations:3d} elapsed={elapsed:7.1f}s "
                f"loss={loss:.8f} x=[{detail}]"
            )
        return loss

    result = minimize(
        objective,
        initial,
        method="Powell",
        bounds=bounds,
        options={
            "maxfev": max_evaluations,
            "xtol": 1e-3,
            "ftol": 1e-5,
        },
    )
    print(
        f"[{name}] done loss={result.fun:.8f} success={result.success} "
        f"n={result.nfev} x={result.x}"
    )
    return result


def _run_metric_rows(
    simulator: FixedBaseStatefulJointwisePdReplay,
    runs: dict[str, tuple[RunData, ...]],
    config: Joint34FitConfig,
    condition: GainCondition,
    kp: dict[str, float],
    kd: dict[str, float],
    *,
    save_plots_to: Path | None,
) -> list[dict[str, Any]]:
    parameters = _parameters(config, condition, kp, kd)
    rows = []
    for trajectory, trajectory_runs in runs.items():
        for run in trajectory_runs:
            replay = simulator.run(
                run,
                delay_sec=config.fixed_delay_sec,
                parameters=parameters,
            )
            joints = (
                config.target_joints
                if trajectory == "static_hold"
                else (run.target_joint,)
            )
            for joint in joints:
                index = MUJOCO_DOF_ORDER.index(joint)
                mask = run.state_mask.copy()
                if trajectory == "static_hold":
                    start = round(
                        len(mask)
                        * (1.0 - config.loss["static_tail_fraction"])
                    )
                    mask[:start] = False
                error = replay.q_rad[mask, index] - run.q_real_rad[mask, index]
                dq_error = (
                    replay.dq_rad_s[mask, index]
                    - run.dq_real_rad_s[mask, index]
                )
                row = {
                    "condition": condition.name,
                    "trajectory": trajectory,
                    "joint": joint,
                    "repeat": run.repeat_index,
                    "run_dir": str(run.run_dir),
                    "q_mae_rad": float(np.mean(np.abs(error))),
                    "q_rmse_rad": float(np.sqrt(np.mean(error**2))),
                    "dq_mae_rad_s": float(np.mean(np.abs(dq_error))),
                    "peak_abs_tau_nm": float(
                        np.max(np.abs(replay.tau_nm[:, index]))
                    ),
                    "torque_saturation_fraction": float(
                        np.mean(
                            np.abs(replay.tau_nm[:, index])
                            >= config.replay.torque_limit_nm - 1e-9
                        )
                    ),
                }
                rows.append(row)
                if save_plots_to is not None:
                    _save_plot(
                        save_plots_to
                        / f"{condition.name}_{trajectory}_{joint}_repeat{run.repeat_index}.png",
                        run,
                        replay,
                        joint,
                        condition,
                        kp,
                        kd,
                    )
    return rows


def _save_plot(
    path: Path,
    run: RunData,
    replay: EquivalentReplayResult,
    joint: str,
    condition: GainCondition,
    kp: dict[str, float],
    kd: dict[str, float],
) -> None:
    index = MUJOCO_DOF_ORDER.index(joint)
    time_axis = np.arange(len(run.q_cmd_rad)) / run.command_rate_hz
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(time_axis, run.q_cmd_rad[:, index], "k--", lw=1.0, label="q_cmd")
    axes[0].plot(
        time_axis[run.state_mask],
        run.q_real_rad[run.state_mask, index],
        lw=1.1,
        label="real",
    )
    axes[0].plot(time_axis, replay.q_rad[:, index], color="tab:red", lw=1.0, label="sim")
    axes[0].set_ylabel("joint angle [rad]")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        time_axis[run.state_mask],
        run.dq_real_rad_s[run.state_mask, index],
        lw=1.0,
        label="real dq",
    )
    axes[1].plot(time_axis, replay.dq_rad_s[:, index], color="tab:red", lw=1.0, label="sim dq")
    axes[1].set(xlabel="time [s]", ylabel="velocity [rad/s]")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    group = next(name for name, joints in _PLOT_GROUPS.items() if joint in joints)
    axes[0].set_title(
        f"{condition.name} {run.metadata['experiment_type']} {joint} / "
        f"group Kp(P350)={kp[group]:.4f}, Kd={kd[group]:.4f}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# _save_plot은 설정 객체를 전역에 숨기지 않도록 identify 진입 시 갱신한다.
_PLOT_GROUPS: dict[str, tuple[str, str]] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for trajectory in sorted({row["trajectory"] for row in rows}):
        selected = [row for row in rows if row["trajectory"] == trajectory]
        result[trajectory] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in (
                "q_mae_rad",
                "q_rmse_rad",
                "dq_mae_rad_s",
                "peak_abs_tau_nm",
                "torque_saturation_fraction",
            )
        }
    return result


def identify_joint34_pd(
    campaigns: dict[str, Path],
    config: Joint34FitConfig,
    output_root: str | Path,
) -> Path:
    conditions = (config.fit_condition,) + config.validation_conditions
    missing = [item.name for item in conditions if item.name not in campaigns]
    if missing:
        raise ValueError(f"campaign 인자가 누락됐습니다: {missing}")
    loaded = {
        item.name: _load_campaign(campaigns[item.name], config, item)
        for item in conditions
    }
    simulator = FixedBaseStatefulJointwisePdReplay(config.replay)
    fit_runs = loaded[config.fit_condition.name].fit
    names = config.group_names

    kp_initial = np.asarray([config.initial_kp[name] for name in names])
    kp_bounds = tuple(config.kp_bounds[name] for name in names)
    kd_initial = np.asarray([config.initial_kd[name] for name in names])
    kd_bounds = tuple(config.kd_bounds[name] for name in names)

    kp_result = _optimize_stage(
        "Kp quasi-static",
        kp_initial,
        kp_bounds,
        config.optimizer["kp_max_evaluations"],
        lambda values: _evaluate_quasistatic(
            simulator,
            fit_runs,
            config,
            config.fit_condition,
            _dict_from_vector(names, values),
            config.initial_kd,
        )[0],
    )
    kp_stage = _dict_from_vector(names, kp_result.x)
    kd_result = _optimize_stage(
        "Kd dynamic",
        kd_initial,
        kd_bounds,
        config.optimizer["kd_max_evaluations"],
        lambda values: _evaluate_dynamic(
            simulator,
            fit_runs,
            config,
            config.fit_condition,
            kp_stage,
            _dict_from_vector(names, values),
        )[0],
    )
    kd_stage = _dict_from_vector(names, kd_result.x)

    final_initial = np.asarray(
        [kp_stage[name] for name in names]
        + [kd_stage[name] for name in names]
    )
    final_bounds = kp_bounds + kd_bounds

    def final_objective(values):
        kp = _dict_from_vector(names, values[:2])
        kd = _dict_from_vector(names, values[2:])
        quasi, _ = _evaluate_quasistatic(
            simulator,
            fit_runs,
            config,
            config.fit_condition,
            kp,
            kd,
        )
        dynamic, _ = _evaluate_dynamic(
            simulator,
            fit_runs,
            config,
            config.fit_condition,
            kp,
            kd,
        )
        return _weighted_mean(
            (
                (quasi, config.loss["quasistatic_final_weight"]),
                (dynamic, config.loss["dynamic_final_weight"]),
            )
        )

    final_result = _optimize_stage(
        "joint refinement",
        final_initial,
        final_bounds,
        config.optimizer["final_max_evaluations"],
        final_objective,
    )
    best_kp = _dict_from_vector(names, final_result.x[:2])
    best_kd = _dict_from_vector(names, final_result.x[2:])
    best = Joint34Candidate(
        kp_at_reference=best_kp,
        kd=best_kd,
        loss=float(final_result.fun),
        success=bool(final_result.success),
        evaluations=int(final_result.nfev),
    )

    output = (
        Path(output_root).expanduser().resolve()
        / f"{datetime.now():%Y%m%d_%H%M%S}_joint34_pd"
    )
    output.mkdir(parents=True, exist_ok=False)
    global _PLOT_GROUPS
    _PLOT_GROUPS = dict(config.groups)

    metrics: dict[str, Any] = {}
    # P350은 fit과 repeat-3 validation을 모두 남긴다.
    fit_rows = _run_metric_rows(
        simulator,
        loaded[config.fit_condition.name].fit,
        config,
        config.fit_condition,
        best_kp,
        best_kd,
        save_plots_to=None,
    )
    validation_rows = _run_metric_rows(
        simulator,
        loaded[config.fit_condition.name].validation,
        config,
        config.fit_condition,
        best_kp,
        best_kd,
        save_plots_to=output,
    )
    metrics[config.fit_condition.name] = {
        "fit_rows": fit_rows,
        "fit_mean": _mean_metrics(fit_rows),
        "validation_rows": validation_rows,
        "validation_mean": _mean_metrics(validation_rows),
    }
    # P850은 어떤 피팅에도 사용하지 않은 외삽 검증이다.
    for condition in config.validation_conditions:
        rows = _run_metric_rows(
            simulator,
            loaded[condition.name].validation,
            config,
            condition,
            best_kp,
            best_kd,
            save_plots_to=output,
        )
        metrics[condition.name] = {
            "validation_rows": rows,
            "validation_mean": _mean_metrics(rows),
        }

    full_p350 = _parameters(
        config,
        config.fit_condition,
        best_kp,
        best_kd,
    )
    params: dict[str, Any] = {
        "model": "joint_group_stateful_backlash_equivalent_pd",
        "fit_condition": config.fit_condition.name,
        "fit_repeats": [1, 2],
        "validation_repeat": 3,
        "fixed": {
            "delay_sec": config.fixed_delay_sec,
            "backlash_total_rad": config.fixed_backlash_total_rad,
            "coulomb_friction_nm": config.fixed_coulomb_friction_nm,
            "position_quantization_rad": config.position_quantization_rad,
            "torque_limit_nm": config.replay.torque_limit_nm,
        },
        "groups": {},
        "recommended_p350": {
            "kp_by_joint_mujoco_order": {
                name: float(value)
                for name, value in zip(MUJOCO_DOF_ORDER, full_p350.kp_eff)
            },
            "kd_by_joint_mujoco_order": {
                name: float(value)
                for name, value in zip(MUJOCO_DOF_ORDER, full_p350.kd_eff)
            },
        },
    }
    for group_name in names:
        params["groups"][group_name] = {
            "joints": list(config.groups[group_name]),
            "kp_at_p350": best_kp[group_name],
            "kd": best_kd[group_name],
            "condition_kp": {
                condition.name: best_kp[group_name]
                * condition.register_p
                / config.reference_register_p
                for condition in conditions
            },
        }
    (output / "params_joint34_pd.yaml").write_text(
        yaml.safe_dump(params, sort_keys=False, allow_unicode=True)
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "optimization.json").write_text(
        json.dumps(
            {
                "kp_stage": {
                    "x": kp_result.x.tolist(),
                    "loss": float(kp_result.fun),
                    "success": bool(kp_result.success),
                    "evaluations": int(kp_result.nfev),
                },
                "kd_stage": {
                    "x": kd_result.x.tolist(),
                    "loss": float(kd_result.fun),
                    "success": bool(kd_result.success),
                    "evaluations": int(kd_result.nfev),
                },
                "final": {
                    "x": final_result.x.tolist(),
                    "loss": best.loss,
                    "success": best.success,
                    "evaluations": best.evaluations,
                },
            },
            indent=2,
        )
        + "\n"
    )
    manifest = {
        "schema_version": 1,
        "fit_config": str(config.source),
        "fit_config_sha256": _sha256(config.source),
        "model_xml": str(config.replay.model_xml.expanduser().resolve()),
        "model_xml_sha256": _sha256(config.replay.model_xml.expanduser().resolve()),
        "campaigns": {
            name: str(Path(path).expanduser().resolve())
            for name, path in campaigns.items()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    report = [
        "# Jandi joint 3/4 equivalent PD identification",
        "",
        "## Result",
        "",
        f"- final fit loss: {best.loss:.8f}",
    ]
    for group_name in names:
        report.append(
            f"- {group_name} ({', '.join(config.groups[group_name])}): "
            f"P350 Kp={best_kp[group_name]:.6f}, Kd={best_kd[group_name]:.6f}"
        )
    report += [
        "",
        "## Fixed parameters",
        "",
        f"- joints 1/2/5/6: Kp={config.fixed_base_kp_at_reference:.6f}, Kd={config.fixed_base_kd:.6f}",
        f"- delay: {config.fixed_delay_sec * 1000.0:.3f} ms",
        f"- backlash total width: {config.fixed_backlash_total_rad:.6f} rad",
        f"- joint Coulomb friction: {config.fixed_coulomb_friction_nm:.6f} Nm",
        "",
        "## Validation means",
        "",
        "| condition | split | trajectory | q MAE deg | q RMSE deg | dq MAE rad/s | peak tau Nm | saturation |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for condition_name, condition_metrics in metrics.items():
        for split_name in ("fit_mean", "validation_mean"):
            if split_name not in condition_metrics:
                continue
            for trajectory, row in condition_metrics[split_name].items():
                report.append(
                    f"| {condition_name} | {split_name.removesuffix('_mean')} | {trajectory} | "
                    f"{np.degrees(row['q_mae_rad']):.4f} | "
                    f"{np.degrees(row['q_rmse_rad']):.4f} | "
                    f"{row['dq_mae_rad_s']:.5f} | "
                    f"{row['peak_abs_tau_nm']:.4f} | "
                    f"{row['torque_saturation_fraction']:.6f} |"
                )
    report += [
        "",
        "## Model contract",
        "",
        "- P350 repeat/seed 1·2만 최적화에 사용했습니다.",
        "- P350 repeat/seed 3과 P850 전체는 최적화에서 제외한 validation입니다.",
        "- RL3/LL3은 한 PD를, RL4/LL4는 한 PD를 공유합니다.",
        "- 1·2·5·6번 PD와 delay/backlash/friction/tick은 고정했습니다.",
        "- Kp는 triangle·static hold·step plateau로 먼저 맞췄습니다.",
        "- Kd는 step transient·multisine으로 먼저 맞췄습니다.",
        "- 최종 단계에서 네 PD 값만 공동 미세조정했습니다.",
        "- replay는 Jandi 학습 actuator와 같은 encoder tick 및 stateful play operator를 사용합니다.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    return output
