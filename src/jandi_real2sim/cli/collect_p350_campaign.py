from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from jandi_real2sim.cli.common import DEFAULT_CONFIG, PROJECT_ROOT
from jandi_real2sim.cli.measurement_common import validate_samples
from jandi_real2sim.config import MUJOCO_DOF_ORDER, RobotConfig, load_robot_config
from jandi_real2sim.identification.dataset import load_run
from jandi_real2sim.trajectory import compact_joint_steps


@dataclass(frozen=True)
class CampaignJob:
    joint: str
    repeat_index: int
    amplitudes_rad: tuple[float, ...]

    @property
    def split_role(self) -> str:
        return "fit" if self.repeat_index in (1, 2) else "validation"


def build_jobs(
    joints: tuple[str, ...],
    repeats: tuple[int, ...],
    general_amplitudes: tuple[float, ...],
    joint2_amplitudes: tuple[float, ...],
) -> tuple[CampaignJob, ...]:
    # repeat를 바깥 순서로 둬 같은 관절의 validation이 바로 이어지는 열 편향을 줄인다.
    return tuple(
        CampaignJob(
            joint=joint,
            repeat_index=repeat,
            amplitudes_rad=(
                joint2_amplitudes if joint in ("RL2_joint", "LL2_joint")
                else general_amplitudes
            ),
        )
        for repeat in repeats
        for joint in joints
    )


def validate_jobs(
    config: RobotConfig,
    jobs: tuple[CampaignJob, ...],
    hold_sec: float,
) -> None:
    pose = config.walking_pose()
    for job in jobs:
        samples = list(
            compact_joint_steps(
                pose,
                job.joint,
                job.amplitudes_rad,
                hold_sec,
                config.bus.command_rate_hz,
            )
        )
        validate_samples(config, samples)


def _metadata_matches(path: Path, job: CampaignJob, p_gain: int) -> bool:
    try:
        metadata = json.loads((path / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    requested = metadata.get("requested_position_pid", {})
    completed = bool(
        metadata.get("valid_flag")
        or metadata.get("data_kind") == "dry_run_plan"
    )
    return bool(
        completed
        and metadata.get("experiment_type") == "compact_step"
        and metadata.get("joint") == job.joint
        and int(metadata.get("repeat_index", -1)) == job.repeat_index
        and tuple(metadata.get("amplitudes_rad", ())) == job.amplitudes_rad
        and requested.get("p") == p_gain
        and requested.get("i") == 0
        and requested.get("d") == 0
    )


def _find_completed(run_root: Path, job: CampaignJob, p_gain: int) -> Path | None:
    matches = [path for path in run_root.iterdir() if path.is_dir() and _metadata_matches(path, job, p_gain)]
    if len(matches) > 1:
        raise RuntimeError(
            f"동일 job의 valid run이 둘 이상입니다. 자동으로 최신을 고르지 않습니다: {matches}"
        )
    return matches[0] if matches else None


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_manifest(
    path: Path,
    campaign_id: str,
    run_root: Path,
    completed: dict[tuple[str, int], Path],
    p_gain: int,
) -> None:
    runs = {
        joint: {
            repeat: completed[(joint, repeat)].name
            for repeat in (1, 2, 3)
        }
        for joint in MUJOCO_DOF_ORDER
    }
    manifest = {
        "campaign_id": campaign_id,
        "data_root": str(run_root.resolve()),
        "position_pid": {"p": p_gain, "i": 0, "d": 0},
        "runs": runs,
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P350 다진폭 compact-step 12관절 × repeat 1~3 자동 수집",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--position-p-gain", type=int, default=350)
    parser.add_argument("--hold-sec", type=float, default=1.0)
    parser.add_argument(
        "--general-amplitudes-rad",
        type=float,
        nargs="+",
        default=(0.02, 0.04, 0.07, 0.10),
    )
    parser.add_argument(
        "--joint2-amplitudes-rad",
        type=float,
        nargs="+",
        default=(0.015, 0.03, 0.05, 0.07),
        help="RL2/LL2의 ±0.08 rad 좁은 쪽 한계를 위한 별도 진폭",
    )
    parser.add_argument(
        "--joints", nargs="+", choices=MUJOCO_DOF_ORDER, default=MUJOCO_DOF_ORDER
    )
    parser.add_argument(
        "--repeats", nargs="+", type=int, choices=(1, 2, 3), default=(1, 2, 3)
    )
    parser.add_argument(
        "--campaign-id",
        default=datetime.now().strftime("P350_multiamp_A_%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="같은 campaign-id에서 이미 완료된 joint/repeat는 이름으로 확인 후 건너뜀",
    )
    args = parser.parse_args()
    if args.execute and args.confirm != "MOVE_JANDI_P350_CAMPAIGN":
        raise SystemExit(
            "전체 자동 수집에는 --execute --confirm MOVE_JANDI_P350_CAMPAIGN이 필요합니다."
        )
    if args.position_p_gain != 350:
        raise SystemExit("이 명령은 P350 campaign 전용입니다: --position-p-gain 350")

    config = load_robot_config(args.config)
    jobs = build_jobs(
        tuple(args.joints),
        tuple(sorted(set(args.repeats))),
        tuple(args.general_amplitudes_rad),
        tuple(args.joint2_amplitudes_rad),
    )
    validate_jobs(config, jobs, args.hold_sec)
    parent = PROJECT_ROOT / ("data/raw/P350" if args.execute else "data/plans/P350")
    campaign_root = parent / args.campaign_id
    run_root = campaign_root / "runs"
    if campaign_root.exists() and not args.resume:
        raise SystemExit(
            f"campaign 폴더가 이미 있습니다: {campaign_root}\n"
            "이어하려면 같은 --campaign-id와 --resume을 사용하세요."
        )
    run_root.mkdir(parents=True, exist_ok=True)

    estimated_sec = sum(
        config.experiment.transition_sec
        + (1 + 4 * len(job.amplitudes_rad)) * args.hold_sec
        for job in jobs
    )
    status_path = campaign_root / "campaign_status.json"
    status: dict[str, Any] = {
        "campaign_id": args.campaign_id,
        "execute": args.execute,
        "position_pid": {"p": 350, "i": 0, "d": 0},
        "job_count": len(jobs),
        "estimated_motion_sec_excluding_io": estimated_sec,
        "completed": [],
    }
    completed: dict[tuple[str, int], Path] = {}
    for index, job in enumerate(jobs, start=1):
        previous = _find_completed(run_root, job, args.position_p_gain)
        if previous is not None:
            print(f"[{index:02d}/{len(jobs)}] SKIP {job.joint} repeat {job.repeat_index}: {previous.name}")
            completed[(job.joint, job.repeat_index)] = previous
            continue
        print(
            f"[{index:02d}/{len(jobs)}] RUN  {job.joint} repeat {job.repeat_index} "
            f"amplitudes={job.amplitudes_rad}"
        )
        before = {path for path in run_root.iterdir() if path.is_dir()}
        command = [
            sys.executable,
            "-m",
            "jandi_real2sim.cli.collect_step",
            job.joint,
            "--config",
            str(config.source),
            "--pose-id",
            "A",
            "--amplitudes-rad",
            *(str(value) for value in job.amplitudes_rad),
            "--hold-sec",
            str(args.hold_sec),
            "--repeat-index",
            str(job.repeat_index),
            "--position-p-gain",
            "350",
            "--position-i-gain",
            "0",
            "--position-d-gain",
            "0",
        ]
        if args.execute:
            command.extend(
                (
                    "--raw-output-dir",
                    str(run_root),
                    "--execute",
                    "--confirm",
                    "MOVE_JANDI",
                )
            )
        else:
            command.extend(("--output-dir", str(run_root)))
        subprocess.run(command, check=True)
        created = {path for path in run_root.iterdir() if path.is_dir()} - before
        if len(created) != 1:
            raise RuntimeError(f"새 run 폴더가 정확히 하나가 아닙니다: {created}")
        run_dir = created.pop()
        if args.execute:
            run = load_run(run_dir)
            settings = run.metadata["actuator_settings"]
            wrong = {
                motor_id: values
                for motor_id, values in settings.items()
                if values["position_p_gain"] != 350
                or values["position_i_gain"] != 0
                or values["position_d_gain"] != 0
            }
            if wrong:
                raise RuntimeError(f"P350/I0/D0 검증 실패: {wrong}")
        completed[(job.joint, job.repeat_index)] = run_dir
        status["completed"] = [
            {
                "joint": joint,
                "repeat_index": repeat,
                "run_dir": str(path),
            }
            for (joint, repeat), path in completed.items()
        ]
        _write_status(status_path, status)

    full_campaign = (
        tuple(args.joints) == MUJOCO_DOF_ORDER
        and tuple(sorted(set(args.repeats))) == (1, 2, 3)
    )
    if full_campaign and args.execute:
        _write_manifest(
            campaign_root / "campaign_manifest.yaml",
            args.campaign_id,
            run_root,
            completed,
            args.position_p_gain,
        )
    print(f"Campaign 완료: {campaign_root}")
    if full_campaign and args.execute:
        print(f"Manifest: {campaign_root / 'campaign_manifest.yaml'}")


if __name__ == "__main__":
    main()
