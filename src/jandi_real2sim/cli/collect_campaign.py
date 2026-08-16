from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from jandi_real2sim.cli.common import DEFAULT_CONFIG, PROJECT_ROOT
from jandi_real2sim.cli.measurement_common import max_command_speed, validate_samples
from jandi_real2sim.config import MUJOCO_DOF_ORDER, RobotConfig, load_robot_config
from jandi_real2sim.trajectory import (
    compact_joint_steps,
    hold_pose,
    multisine_joint,
    triangle_joint,
)


@dataclass(frozen=True)
class SpecJob:
    experiment_name: str
    experiment_type: str
    joint: str | None
    run_index: int
    split_role: str
    parameters: dict[str, Any]

    @property
    def key(self) -> str:
        joint = self.joint or "all_joints"
        return f"{self.experiment_name}/{joint}/{self.run_index}/{self.split_role}"


def _joints(value: Any) -> tuple[str, ...]:
    joints = MUJOCO_DOF_ORDER if value == "all" else tuple(value)
    if not joints or any(joint not in MUJOCO_DOF_ORDER for joint in joints):
        raise ValueError(f"잘못된 joints: {value}")
    if len(set(joints)) != len(joints):
        raise ValueError(f"중복 joints: {value}")
    return joints


def _joint_value(spec: Any, joint: str) -> Any:
    if isinstance(spec, dict) and "default" in spec:
        return spec.get("per_joint", {}).get(joint, spec["default"])
    return spec


def resolve_pid(raw: dict[str, Any]) -> dict[str, dict[str, int]]:
    default = raw["default"]
    overrides = raw.get("overrides", {})
    unknown = set(overrides) - set(MUJOCO_DOF_ORDER)
    if unknown:
        raise ValueError(f"PID overrides의 알 수 없는 관절: {unknown}")
    result: dict[str, dict[str, int]] = {}
    for joint in MUJOCO_DOF_ORDER:
        values = overrides.get(joint, default)
        if set(values) != {"p", "i", "d"}:
            raise ValueError(f"{joint}: PID는 p/i/d가 모두 필요합니다.")
        converted = {name: int(value) for name, value in values.items()}
        if any(not 0 <= value <= 16383 for value in converted.values()):
            raise ValueError(f"{joint}: PID 범위 [0,16383] 초과: {converted}")
        result[joint] = converted
    return result


def build_spec_jobs(experiments: list[dict[str, Any]]) -> tuple[SpecJob, ...]:
    jobs: list[SpecJob] = []
    names: set[str] = set()
    for experiment in experiments:
        name = str(experiment["name"])
        if name in names:
            raise ValueError(f"중복 experiment name: {name}")
        names.add(name)
        if not experiment.get("enabled", False):
            continue
        kind = str(experiment["type"])
        if kind == "compact_step":
            joints = _joints(experiment["joints"])
            repeats = tuple(int(value) for value in experiment["repeats"])
            for repeat in repeats:
                for joint in joints:
                    jobs.append(
                        SpecJob(
                            name,
                            kind,
                            joint,
                            repeat,
                            "fit" if repeat in (1, 2) else "validation",
                            {
                                "amplitudes_rad": tuple(
                                    float(value)
                                    for value in _joint_value(
                                        experiment["amplitudes_rad"], joint
                                    )
                                ),
                                "hold_sec": float(experiment["hold_sec"]),
                            },
                        )
                    )
        elif kind == "triangle":
            joints = _joints(experiment["joints"])
            for repeat in tuple(int(value) for value in experiment["repeats"]):
                for joint in joints:
                    jobs.append(
                        SpecJob(
                            name,
                            kind,
                            joint,
                            repeat,
                            "diagnostic",
                            {
                                "amplitude_rad": float(
                                    _joint_value(experiment["amplitude_rad"], joint)
                                ),
                                "frequency_hz": float(experiment["frequency_hz"]),
                                "cycles": int(experiment["cycles"]),
                                "max_command_speed_rad_s": float(
                                    experiment["max_command_speed_rad_s"]
                                ),
                            },
                        )
                    )
        elif kind == "multisine":
            joints = _joints(experiment["joints"])
            for seed_spec in experiment["seeds"]:
                for joint in joints:
                    jobs.append(
                        SpecJob(
                            name,
                            kind,
                            joint,
                            int(seed_spec["seed"]),
                            str(seed_spec["split_role"]),
                            {
                                "amplitude_rad": float(
                                    _joint_value(experiment["amplitude_rad"], joint)
                                ),
                                "frequencies_hz": tuple(
                                    float(value) for value in experiment["frequencies_hz"]
                                ),
                                "duration_sec": float(experiment["duration_sec"]),
                                "fade_sec": float(experiment["fade_sec"]),
                                "max_command_speed_rad_s": float(
                                    experiment["max_command_speed_rad_s"]
                                ),
                            },
                        )
                    )
        elif kind == "static_hold":
            for repeat in tuple(int(value) for value in experiment["repeats"]):
                jobs.append(
                    SpecJob(
                        name,
                        kind,
                        None,
                        repeat,
                        "baseline",
                        {"duration_sec": float(experiment["duration_sec"])},
                    )
                )
        else:
            raise ValueError(f"지원하지 않는 experiment type: {kind}")
    if not jobs:
        raise ValueError("enabled=true인 실험이 없습니다.")
    if len({job.key for job in jobs}) != len(jobs):
        raise ValueError("실험 job key가 중복됩니다. repeat/seed를 확인하세요.")
    return tuple(jobs)


def validate_spec_and_jobs(
    raw: dict[str, Any], config: RobotConfig, jobs: tuple[SpecJob, ...]
) -> None:
    if int(raw["schema_version"]) != 1:
        raise ValueError("지원하는 collection spec schema_version은 1입니다.")
    logging = raw["logging"]
    expected_logging = {
        "record_transition": False,
        "command_rate_hz": config.bus.command_rate_hz,
        "state_read_rate_hz": config.bus.state_read_rate_hz,
        "hardware_error_read_rate_hz": config.bus.hardware_error_read_rate_hz,
        "fields": "full_actuator_state",
    }
    if logging != expected_logging:
        raise ValueError(
            "logging 계약은 데이터 호환성을 위해 고정입니다: "
            f"expected={expected_logging}, actual={logging}"
        )
    safety = raw["safety"]
    required_safety = {
        "max_temperature_c",
        "min_input_voltage_v",
        "max_abs_current_a",
        "max_abs_pwm_percent",
        "max_abs_position_error_rad",
        "consecutive_state_samples",
        "between_jobs_sec",
        "cooldown_every_executed_jobs",
        "cooldown_sec",
    }
    if set(safety) != required_safety:
        raise ValueError(
            f"safety 항목 불일치: expected={required_safety}, actual={set(safety)}"
        )
    if int(safety["consecutive_state_samples"]) < 1:
        raise ValueError("consecutive_state_samples는 1 이상이어야 합니다.")
    if int(safety["cooldown_every_executed_jobs"]) < 1:
        raise ValueError("cooldown_every_executed_jobs는 1 이상이어야 합니다.")
    if float(safety["between_jobs_sec"]) < 0 or float(safety["cooldown_sec"]) < 0:
        raise ValueError("cooldown 시간은 0 이상이어야 합니다.")
    for name in (
        "max_temperature_c",
        "min_input_voltage_v",
        "max_abs_current_a",
        "max_abs_pwm_percent",
        "max_abs_position_error_rad",
    ):
        if float(safety[name]) <= 0:
            raise ValueError(f"{name}은 0보다 커야 합니다.")
    pose = config.walking_pose()
    for job in jobs:
        if job.experiment_type == "compact_step":
            samples = list(
                compact_joint_steps(
                    pose,
                    str(job.joint),
                    job.parameters["amplitudes_rad"],
                    job.parameters["hold_sec"],
                    config.bus.command_rate_hz,
                )
            )
        elif job.experiment_type == "triangle":
            samples = list(
                triangle_joint(
                    pose,
                    str(job.joint),
                    job.parameters["amplitude_rad"],
                    job.parameters["frequency_hz"],
                    job.parameters["cycles"],
                    config.bus.command_rate_hz,
                )
            )
        elif job.experiment_type == "multisine":
            frequencies = job.parameters["frequencies_hz"]
            if not 4 <= len(frequencies) <= 6:
                raise ValueError(f"{job.key}: multisine 주파수는 4~6개여야 합니다.")
            if job.parameters["duration_sec"] < 5.0 / min(frequencies):
                raise ValueError(f"{job.key}: 최저주파수 5주기보다 duration이 짧습니다.")
            samples = list(
                multisine_joint(
                    pose,
                    str(job.joint),
                    job.parameters["amplitude_rad"],
                    frequencies,
                    job.parameters["duration_sec"],
                    job.run_index,
                    config.bus.command_rate_hz,
                    fade_sec=job.parameters["fade_sec"],
                )
            )
        else:
            samples = list(
                hold_pose(
                    pose,
                    job.parameters["duration_sec"],
                    config.bus.command_rate_hz,
                )
            )
        validate_samples(config, samples)
        if job.joint and job.experiment_type in ("triangle", "multisine"):
            speed = max_command_speed(samples, config.bus.command_rate_hz, job.joint)
            if speed > job.parameters["max_command_speed_rad_s"] + 1e-12:
                raise ValueError(
                    f"{job.key}: command speed {speed:.6f}가 제한 "
                    f"{job.parameters['max_command_speed_rad_s']:.6f}을 초과합니다."
                )


def _job_command(
    job: SpecJob,
    config: RobotConfig,
    pose_id: str,
    pid_json: Path,
    safety_json: Path,
    run_root: Path,
    execute: bool,
) -> list[str]:
    module = {
        "compact_step": "jandi_real2sim.cli.collect_step",
        "triangle": "jandi_real2sim.cli.collect_triangle",
        "multisine": "jandi_real2sim.cli.collect_multisine",
        "static_hold": "jandi_real2sim.cli.collect_hold",
    }[job.experiment_type]
    command = [
        sys.executable, "-m", module,
        "--config", str(config.source),
        "--pose-id", pose_id,
        "--position-pid-json", str(pid_json),
        "--campaign-experiment-name", job.experiment_name,
        "--safety-json", str(safety_json),
    ]
    if job.joint:
        command.append(job.joint)
    if job.experiment_type == "compact_step":
        command.extend(("--amplitudes-rad", *(str(v) for v in job.parameters["amplitudes_rad"])))
        command.extend(("--hold-sec", str(job.parameters["hold_sec"]), "--repeat-index", str(job.run_index)))
    elif job.experiment_type == "triangle":
        command.extend((
            "--amplitude-rad", str(job.parameters["amplitude_rad"]),
            "--frequency-hz", str(job.parameters["frequency_hz"]),
            "--cycles", str(job.parameters["cycles"]),
            "--max-command-speed-rad-s", str(job.parameters["max_command_speed_rad_s"]),
            "--repeat-index", str(job.run_index),
        ))
    elif job.experiment_type == "multisine":
        command.extend(("--amplitude-rad", str(job.parameters["amplitude_rad"])))
        command.extend(("--frequencies-hz", *(str(v) for v in job.parameters["frequencies_hz"])))
        command.extend((
            "--duration-sec", str(job.parameters["duration_sec"]),
            "--fade-sec", str(job.parameters["fade_sec"]),
            "--seed", str(job.run_index),
            "--split-role", job.split_role,
            "--max-command-speed-rad-s", str(job.parameters["max_command_speed_rad_s"]),
        ))
    else:
        command.extend(("--duration-sec", str(job.parameters["duration_sec"]), "--repeat-index", str(job.run_index)))
    if execute:
        command.extend(("--raw-output-dir", str(run_root), "--execute", "--confirm", "MOVE_JANDI"))
    else:
        command.extend(("--output-dir", str(run_root)))
    return command


def _metadata_matches(path: Path, job: SpecJob) -> bool:
    try:
        metadata = json.loads((path / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    completed = metadata.get("valid_flag") or metadata.get("data_kind") == "dry_run_plan"
    if not completed or metadata.get("campaign_experiment_name") != job.experiment_name:
        return False
    if metadata.get("experiment_type") != job.experiment_type:
        return False
    if job.joint is not None and metadata.get("joint") != job.joint:
        return False
    if job.experiment_type == "multisine":
        return int(metadata.get("seed", -1)) == job.run_index and metadata.get("split_role") == job.split_role
    return int(metadata.get("repeat_index", -1)) == job.run_index


def _find_completed(run_root: Path, job: SpecJob) -> Path | None:
    matches = [path for path in run_root.iterdir() if path.is_dir() and _metadata_matches(path, job)]
    if len(matches) > 1:
        raise RuntimeError(f"{job.key}: valid run이 둘 이상이라 자동 선택하지 않습니다: {matches}")
    return matches[0] if matches else None


def _validate_actual_run(
    run_dir: Path, pid: dict[str, dict[str, int]], config: RobotConfig
) -> int:
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if not metadata.get("valid_flag"):
        raise RuntimeError(f"invalid run: {run_dir}: {metadata.get('invalid_reason')}")
    settings = metadata["actuator_settings"]
    for joint in config.joints:
        values = settings[str(joint.motor_id)]
        for name, expected in pid[joint.name].items():
            actual = values[f"position_{name}_gain"]
            if actual != expected:
                raise RuntimeError(
                    f"{run_dir.name}: {joint.name} {name} readback {actual}!={expected}"
                )
    with (run_dir / "telemetry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = int(metadata["expected_sample_count"])
    if len(rows) != expected or [int(row["cycle_index"]) for row in rows] != list(range(expected)):
        raise RuntimeError(f"{run_dir.name}: 표본 수 또는 cycle 불연속")
    errors = [
        int(value)
        for row in rows
        for key, value in row.items()
        if key.endswith("/hardware_error") and value
    ]
    if any(errors):
        raise RuntimeError(f"{run_dir.name}: Hardware Error={errors}")
    return sum(int(row["overrun_ns"]) > 0 for row in rows)


def _save_status(path: Path, status: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YAML 명세 기반 Jandi Real2Sim 자동 데이터 수집"
    )
    parser.add_argument(
        "--spec", type=Path, default=PROJECT_ROOT / "configs" / "collection_campaign.yaml"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.execute and args.confirm != "MOVE_JANDI_CAMPAIGN":
        raise SystemExit("실제 자동 수집에는 --execute --confirm MOVE_JANDI_CAMPAIGN이 필요합니다.")

    spec_path = args.spec.expanduser().resolve()
    raw = yaml.safe_load(spec_path.read_text())
    config = load_robot_config(args.config)
    pid = resolve_pid(raw["position_pid"])
    jobs = build_spec_jobs(raw["experiments"])
    validate_spec_and_jobs(raw, config, jobs)
    campaign = raw["campaign"]
    campaign_id = str(campaign["id"])
    output_group = str(campaign["output_group"])
    if not campaign_id or not output_group or "/" in campaign_id or "/" in output_group:
        raise ValueError("campaign id/output_group은 비어 있지 않은 단일 폴더 이름이어야 합니다.")
    parent = PROJECT_ROOT / ("data/raw" if args.execute else "data/plans") / output_group
    root = parent / campaign_id
    run_root = root / "runs"
    if root.exists() and not args.resume:
        raise SystemExit(f"이미 존재합니다: {root}\n같은 spec으로 이어가려면 --resume을 사용하세요.")
    run_root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "collection_campaign.snapshot.yaml"
    pid_json = root / "resolved_position_pid.json"
    safety_json = root / "resolved_live_safety.json"
    live_safety = {
        key: raw["safety"][key]
        for key in (
            "max_temperature_c",
            "min_input_voltage_v",
            "max_abs_current_a",
            "max_abs_pwm_percent",
            "max_abs_position_error_rad",
            "consecutive_state_samples",
        )
    }
    if not snapshot.exists():
        shutil.copy2(spec_path, snapshot)
        pid_json.write_text(json.dumps(pid, indent=2) + "\n")
        safety_json.write_text(json.dumps(live_safety, indent=2) + "\n")
    else:
        if snapshot.read_bytes() != spec_path.read_bytes():
            raise RuntimeError("resume 중 spec이 변경됐습니다. 새 campaign id를 사용하세요.")
        if json.loads(pid_json.read_text()) != pid:
            raise RuntimeError("resume 중 resolved PID가 변경됐습니다.")
        if not safety_json.exists() or json.loads(safety_json.read_text()) != live_safety:
            raise RuntimeError("resume 중 resolved live safety가 변경됐습니다.")

    status_path = root / "campaign_status.json"
    completed: dict[str, str] = {}
    warnings: dict[str, int] = {}
    executed_jobs = 0
    print(f"Campaign {campaign_id}: {len(jobs)} jobs, output={root}")
    for index, job in enumerate(jobs, start=1):
        previous = _find_completed(run_root, job)
        if previous is not None:
            print(f"[{index:03d}/{len(jobs)}] SKIP {job.key}: {previous.name}")
            completed[job.key] = previous.name
            continue
        print(f"[{index:03d}/{len(jobs)}] RUN  {job.key} params={job.parameters}")
        before = {path for path in run_root.iterdir() if path.is_dir()}
        subprocess.run(
            _job_command(
                job,
                config,
                str(campaign["pose_id"]),
                pid_json,
                safety_json,
                run_root,
                args.execute,
            ),
            check=True,
        )
        created = {path for path in run_root.iterdir() if path.is_dir()} - before
        if len(created) != 1:
            raise RuntimeError(f"새 run 폴더가 정확히 하나가 아닙니다: {created}")
        run_dir = created.pop()
        if args.execute:
            overrun_count = _validate_actual_run(run_dir, pid, config)
            if overrun_count:
                warnings[job.key] = overrun_count
        completed[job.key] = run_dir.name
        executed_jobs += 1
        _save_status(
            status_path,
            {
                "campaign_id": campaign_id,
                "spec_snapshot": str(snapshot),
                "completed": completed,
                "deadline_overrun_samples": warnings,
            },
        )
        if args.execute and index < len(jobs):
            between = float(raw["safety"]["between_jobs_sec"])
            if between > 0:
                print(f"Torque Off job 간 대기: {between:.1f} s")
                time.sleep(between)
            every = int(raw["safety"]["cooldown_every_executed_jobs"])
            if executed_jobs % every == 0:
                cooldown = float(raw["safety"]["cooldown_sec"])
                if cooldown > 0:
                    print(f"Torque Off 정기 냉각: {cooldown:.1f} s ({executed_jobs} jobs)")
                    time.sleep(cooldown)
    print(f"Campaign 완료: {root}")
    print(f"Status: {status_path}")


if __name__ == "__main__":
    main()
