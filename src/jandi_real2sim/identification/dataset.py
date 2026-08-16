from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from jandi_real2sim.config import MUJOCO_DOF_ORDER


@dataclass(frozen=True)
class RunData:
    run_dir: Path
    metadata: dict[str, Any]
    target_joint: str
    repeat_index: int
    split_role: str
    phase: tuple[str, ...]
    q_cmd_rad: np.ndarray
    q_real_rad: np.ndarray
    dq_real_rad_s: np.ndarray
    q_trajectory_rad: np.ndarray
    dq_trajectory_rad_s: np.ndarray
    present_pwm_raw: np.ndarray
    present_current_a: np.ndarray
    input_voltage_v: np.ndarray
    state_mask: np.ndarray
    overrun_mask: np.ndarray
    tx_time_ns: np.ndarray
    rx_time_ns: np.ndarray
    q_init_rad: np.ndarray
    dq_init_rad_s: np.ndarray

    @property
    def target_index(self) -> int:
        return MUJOCO_DOF_ORDER.index(self.target_joint)

    @property
    def command_rate_hz(self) -> int:
        return int(self.metadata["command_rate_hz"])


def _joint_vector(mapping: dict[str, Any], label: str) -> np.ndarray:
    missing = [name for name in MUJOCO_DOF_ORDER if name not in mapping]
    if missing:
        raise ValueError(f"{label}에 관절이 누락됐습니다: {missing}")
    return np.asarray([float(mapping[name]) for name in MUJOCO_DOF_ORDER])


def load_run(
    run_dir: str | Path,
    *,
    target_joint_override: str | None = None,
) -> RunData:
    run_dir = Path(run_dir).expanduser().resolve()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if metadata.get("data_kind") != "real_measurement":
        raise ValueError(f"실측 run이 아닙니다: {run_dir}")
    if not metadata.get("valid_flag"):
        raise ValueError(
            f"invalid run입니다: {run_dir}: {metadata.get('invalid_reason', '')}"
        )
    experiment_type = str(metadata.get("experiment_type"))
    if experiment_type not in {
        "compact_step",
        "triangle",
        "multisine",
        "static_hold",
    }:
        raise ValueError(
            f"지원하지 않는 식별 experiment_type={experiment_type}: {run_dir}"
        )
    metadata_joint = metadata.get("joint")
    if metadata_joint is None and target_joint_override is None:
        raise ValueError(
            f"{experiment_type} run에는 target_joint_override가 필요합니다: "
            f"{run_dir}"
        )
    if (
        metadata_joint is not None
        and target_joint_override is not None
        and str(metadata_joint) != target_joint_override
    ):
        raise ValueError(
            f"metadata joint와 override가 다릅니다: "
            f"{metadata_joint} != {target_joint_override}"
        )
    target_joint = str(
        metadata_joint if metadata_joint is not None else target_joint_override
    )
    if target_joint not in MUJOCO_DOF_ORDER:
        raise ValueError(f"알 수 없는 target joint: {target_joint}")

    with (run_dir / "telemetry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = int(metadata["expected_sample_count"])
    if len(rows) != expected:
        raise ValueError(f"표본 수 불일치: {run_dir}: {len(rows)} != {expected}")
    cycles = [int(row["cycle_index"]) for row in rows]
    if cycles != list(range(expected)):
        raise ValueError(f"cycle_index 불연속: {run_dir}")

    sample_count = len(rows)
    joint_count = len(MUJOCO_DOF_ORDER)
    q_cmd = np.empty((sample_count, joint_count), dtype=np.float64)
    q_real = np.full_like(q_cmd, np.nan)
    dq_real = np.full_like(q_cmd, np.nan)
    q_trajectory = np.full_like(q_cmd, np.nan)
    dq_trajectory = np.full_like(q_cmd, np.nan)
    present_pwm_raw = np.full_like(q_cmd, np.nan)
    present_current_a = np.full_like(q_cmd, np.nan)
    input_voltage_v = np.full_like(q_cmd, np.nan)
    state_mask = np.zeros(sample_count, dtype=bool)
    overrun_mask = np.zeros(sample_count, dtype=bool)
    tx_time = np.empty(sample_count, dtype=np.int64)
    rx_time = np.empty(sample_count, dtype=np.int64)
    phase: list[str] = []
    for sample_index, row in enumerate(rows):
        phase.append(row["phase"])
        overrun_mask[sample_index] = int(row["overrun_ns"]) > 0
        tx_time[sample_index] = int(row["tx_time_ns"])
        rx_time[sample_index] = int(row["rx_time_ns"])
        for joint_index, joint_name in enumerate(MUJOCO_DOF_ORDER):
            q_cmd[sample_index, joint_index] = float(
                row[f"{joint_name}/q_cmd_sent_rad"]
            )
        if row["acquisition_kind"] == "state":
            state_mask[sample_index] = True
            for joint_index, joint_name in enumerate(MUJOCO_DOF_ORDER):
                q_real[sample_index, joint_index] = float(
                    row[f"{joint_name}/q_present_rad"]
                )
                dq_real[sample_index, joint_index] = float(
                    row[f"{joint_name}/dq_present_rad_s"]
                )
                q_trajectory[sample_index, joint_index] = float(
                    row[f"{joint_name}/position_trajectory_rad"]
                )
                dq_trajectory[sample_index, joint_index] = float(
                    row[f"{joint_name}/velocity_trajectory_rad_s"]
                )
                present_pwm_raw[sample_index, joint_index] = float(
                    row[f"{joint_name}/present_pwm_raw"]
                )
                present_current_a[sample_index, joint_index] = float(
                    row[f"{joint_name}/current_A"]
                )
                input_voltage_v[sample_index, joint_index] = float(
                    row[f"{joint_name}/input_voltage_V"]
                )

    hardware_errors = [
        int(row[f"{joint_name}/hardware_error"])
        for row in rows
        for joint_name in MUJOCO_DOF_ORDER
        if row[f"{joint_name}/hardware_error"] != ""
    ]
    if any(hardware_errors):
        raise ValueError(f"Hardware Error가 있는 run입니다: {run_dir}")
    # 누락 없는 고정 cycle 로그의 드문 deadline overrun은 run 전체를 버리지
    # 않고 해당 state 표본만 loss에서 제외한다. 원본 CSV는 수정하지 않는다.
    state_mask &= ~overrun_mask

    return RunData(
        run_dir=run_dir,
        metadata=metadata,
        target_joint=target_joint,
        repeat_index=int(metadata.get("repeat_index", metadata.get("seed"))),
        split_role=str(metadata["split_role"]),
        phase=tuple(phase),
        q_cmd_rad=q_cmd,
        q_real_rad=q_real,
        dq_real_rad_s=dq_real,
        q_trajectory_rad=q_trajectory,
        dq_trajectory_rad_s=dq_trajectory,
        present_pwm_raw=present_pwm_raw,
        present_current_a=present_current_a,
        input_voltage_v=input_voltage_v,
        state_mask=state_mask,
        overrun_mask=overrun_mask,
        tx_time_ns=tx_time,
        rx_time_ns=rx_time,
        q_init_rad=_joint_vector(metadata["q_init_rad"], "q_init_rad"),
        dq_init_rad_s=_joint_vector(metadata["dq_init_rad_s"], "dq_init_rad_s"),
    )


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    source: Path
    data_root: Path
    runs: tuple[RunData, ...]


def load_campaign(path: str | Path) -> Campaign:
    """명시된 폴더 이름만 읽는다. mtime이나 'latest' 선택은 사용하지 않는다."""
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text())
    campaign_id = str(raw["campaign_id"])
    data_root = Path(raw["data_root"]).expanduser().resolve()
    run_mapping = raw["runs"]
    if tuple(run_mapping) != MUJOCO_DOF_ORDER:
        raise ValueError("campaign 관절 순서는 RL1..RL6, LL1..LL6여야 합니다.")
    label_overrides = {
        str(item["directory"]): str(item["reason"])
        for item in raw.get("metadata_label_overrides", [])
    }
    runs: list[RunData] = []
    for joint_name in MUJOCO_DOF_ORDER:
        repeats = run_mapping[joint_name]
        for repeat_index in (1, 2, 3):
            directory = repeats.get(repeat_index, repeats.get(str(repeat_index)))
            if not directory:
                raise ValueError(f"{joint_name} repeat {repeat_index}가 누락됐습니다.")
            run = load_run(data_root / str(directory))
            expected_role = "fit" if repeat_index in (1, 2) else "validation"
            mismatched = (
                run.target_joint != joint_name
                or run.repeat_index != repeat_index
                or run.split_role != expected_role
            )
            if mismatched and str(directory) not in label_overrides:
                raise ValueError(
                    f"manifest/metadata label 불일치: {directory}: "
                    f"metadata=({run.target_joint},{run.repeat_index},{run.split_role}), "
                    f"manifest=({joint_name},{repeat_index},{expected_role})"
                )
            if mismatched:
                metadata = dict(run.metadata)
                metadata["campaign_original_labels"] = {
                    "joint": run.target_joint,
                    "repeat_index": run.repeat_index,
                    "split_role": run.split_role,
                }
                metadata["campaign_label_override_reason"] = label_overrides[
                    str(directory)
                ]
                run = dataclasses.replace(
                    run,
                    metadata=metadata,
                    target_joint=joint_name,
                    repeat_index=repeat_index,
                    split_role=expected_role,
                )
            runs.append(run)

    signatures = {
        json.dumps(
            {
                "pose_id": run.metadata["pose_id"],
                "hold_sec": run.metadata["hold_sec"],
                "command_rate_hz": run.metadata["command_rate_hz"],
                "actuator_settings": run.metadata.get("actuator_settings"),
            },
            sort_keys=True,
        )
        for run in runs
    }
    if len(signatures) != 1:
        raise ValueError("campaign의 자세·hold·rate·액추에이터 설정이 일치하지 않습니다.")
    return Campaign(campaign_id, source, data_root, tuple(runs))


def discover_pair_runs(
    data_root: str | Path,
    target_joints: tuple[str, str],
    pose_id: str,
) -> tuple[RunData, ...]:
    """각 joint/repeat 1~3의 가장 최근 완전한 동일조건 campaign을 고른다."""
    data_root = Path(data_root).expanduser().resolve()
    candidates: list[RunData] = []
    for metadata_path in data_root.glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text())
            if (
                metadata.get("valid_flag")
                and metadata.get("experiment_type") == "compact_step"
                and metadata.get("pose_id") == pose_id
                and metadata.get("joint") in target_joints
                and int(metadata.get("repeat_index", 0)) in (1, 2, 3)
            ):
                candidates.append(load_run(metadata_path.parent))
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            continue
    if not candidates:
        raise ValueError(f"compact step run을 찾지 못했습니다: {data_root}")

    # 진폭·hold·MX 설정이 같은 run끼리만 하나의 campaign으로 묶는다.
    groups: dict[str, list[RunData]] = {}
    for run in candidates:
        signature = json.dumps(
            {
                "amplitudes_rad": run.metadata["amplitudes_rad"],
                "hold_sec": run.metadata["hold_sec"],
                "command_rate_hz": run.metadata["command_rate_hz"],
                "actuator_settings": run.metadata.get("actuator_settings"),
            },
            sort_keys=True,
        )
        groups.setdefault(signature, []).append(run)

    complete_groups: list[tuple[float, tuple[RunData, ...]]] = []
    required = {(joint, repeat) for joint in target_joints for repeat in (1, 2, 3)}
    for runs in groups.values():
        latest: dict[tuple[str, int], RunData] = {}
        for run in runs:
            key = (run.target_joint, run.repeat_index)
            if key not in latest or run.run_dir.stat().st_mtime > latest[key].run_dir.stat().st_mtime:
                latest[key] = run
        if required.issubset(latest):
            selected = tuple(latest[key] for key in sorted(required))
            complete_groups.append(
                (max(run.run_dir.stat().st_mtime for run in selected), selected)
            )
    if not complete_groups:
        found = sorted((run.target_joint, run.repeat_index) for run in candidates)
        raise ValueError(f"동일조건 6-run campaign이 완성되지 않았습니다: found={found}")
    return max(complete_groups, key=lambda item: item[0])[1]


def split_fit_validation(
    runs: tuple[RunData, ...],
) -> tuple[tuple[RunData, ...], tuple[RunData, ...]]:
    expected = {
        (joint, repeat)
        for joint in {run.target_joint for run in runs}
        for repeat in (1, 2, 3)
    }
    actual = {(run.target_joint, run.repeat_index) for run in runs}
    if len(runs) != 6 or len(expected) != 6 or actual != expected:
        raise ValueError(
            "campaign은 두 관절 × repeat 1~3의 고유한 6개 run이어야 합니다: "
            f"actual={sorted(actual)}"
        )
    wrong_roles = [
        (run.target_joint, run.repeat_index, run.split_role)
        for run in runs
        if run.split_role != ("fit" if run.repeat_index in (1, 2) else "validation")
    ]
    if wrong_roles:
        raise ValueError(f"repeat별 split_role이 잘못됐습니다: {wrong_roles}")
    def trajectory_signature(run: RunData) -> dict[str, Any]:
        experiment_type = str(run.metadata["experiment_type"])
        common = {
            "experiment_type": experiment_type,
            "command_rate_hz": run.metadata["command_rate_hz"],
            "pose_id": run.metadata["pose_id"],
            "actuator_settings": run.metadata.get("actuator_settings"),
        }
        if experiment_type == "compact_step":
            common.update(
                amplitudes_rad=run.metadata["amplitudes_rad"],
                hold_sec=run.metadata["hold_sec"],
            )
        elif experiment_type == "triangle":
            common.update(
                amplitude_rad=run.metadata["amplitude_rad"],
                frequency_hz=run.metadata["frequency_hz"],
                cycles=run.metadata["cycles"],
            )
        elif experiment_type == "multisine":
            common.update(
                amplitude_rad=run.metadata["amplitude_rad"],
                frequencies_hz=run.metadata["frequencies_hz"],
                duration_sec=run.metadata["duration_sec"],
                fade_sec=run.metadata["fade_sec"],
            )
        else:
            raise ValueError(f"지원하지 않는 experiment_type: {experiment_type}")
        return common

    signatures = {
        json.dumps(
            trajectory_signature(run),
            sort_keys=True,
        )
        for run in runs
    }
    if len(signatures) != 1:
        raise ValueError("6개 run의 궤적·자세·액추에이터 설정이 서로 다릅니다.")
    fit = tuple(run for run in runs if run.split_role == "fit")
    validation = tuple(run for run in runs if run.split_role == "validation")
    if len(fit) != 4 or len(validation) != 2:
        raise ValueError(
            f"예상 split은 fit 4/validation 2입니다: {len(fit)}/{len(validation)}"
        )
    return fit, validation


def interpolate_state(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    indices = np.arange(len(values))
    return np.interp(indices, indices[mask], values[mask])


def estimate_delay_samples(run: RunData, threshold_fraction: float = 0.05) -> list[float]:
    """각 step edge에서 정상상태 변화의 threshold_fraction에 처음 도달한 지연."""
    joint_index = run.target_index
    command = run.q_cmd_rad[:, joint_index]
    position = interpolate_state(run.q_real_rad[:, joint_index], run.state_mask)
    edge_indices = np.flatnonzero(np.abs(np.diff(command)) > 1e-12) + 1
    period_ns = 1e9 / run.command_rate_hz
    delays: list[float] = []
    for edge_number, edge in enumerate(edge_indices):
        next_edge = (
            int(edge_indices[edge_number + 1])
            if edge_number + 1 < len(edge_indices)
            else len(command)
        )
        before = position[max(0, edge - 50):edge]
        after_state = np.flatnonzero(run.state_mask[edge:next_edge]) + edge
        if not len(before) or not len(after_state):
            continue
        q0 = float(np.median(before))
        q_inf = float(np.median(position[after_state[-50:]]))
        delta = q_inf - q0
        if abs(delta) < 1e-9:
            continue
        threshold = q0 + threshold_fraction * delta
        direction = 1.0 if delta > 0.0 else -1.0
        crossing = next(
            (
                index
                for index in after_state
                if direction * (position[index] - threshold) >= 0.0
            ),
            None,
        )
        if crossing is not None:
            delay_ns = run.rx_time_ns[crossing] - run.tx_time_ns[edge]
            delays.append(float(delay_ns / period_ns))
    return delays
