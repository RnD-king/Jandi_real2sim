from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from jandi_real2sim.config import MUJOCO_DOF_ORDER
from jandi_real2sim.identification.dataset import RunData


@dataclass(frozen=True)
class ReplayConfig:
    model_xml: Path
    physics_dt_sec: float
    torque_limit_nm: float
    kp_baseline: np.ndarray
    kd_baseline: np.ndarray


@dataclass(frozen=True)
class ReplayResult:
    q_rad: np.ndarray
    dq_rad_s: np.ndarray
    tau_nm: np.ndarray
    sample_substep: int


class FixedBaseReplay:
    """Jandi pelvis를 고정하고 12개 관절에 외부 PD 토크를 가한다."""

    def __init__(self, config: ReplayConfig):
        self.config = config
        spec = mujoco.MjSpec.from_file(str(config.model_xml.expanduser().resolve()))
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        freejoint = spec.joint("floating_base_joint")
        if freejoint is None:
            raise ValueError("MJCF에 floating_base_joint가 없습니다.")
        spec.delete(freejoint)
        self.model = spec.compile()
        self.model.opt.timestep = config.physics_dt_sec
        if self.model.nq != len(MUJOCO_DOF_ORDER) or self.model.nv != len(
            MUJOCO_DOF_ORDER
        ):
            raise ValueError(
                f"식별 모델 자유도 불일치: nq={self.model.nq}, nv={self.model.nv}"
            )
        if self.model.nu != 0:
            raise ValueError(f"XML actuator 제거 실패: nu={self.model.nu}")
        self.qpos_indices = np.asarray(
            [self.model.joint(name).qposadr[0] for name in MUJOCO_DOF_ORDER]
        )
        self.dof_indices = np.asarray(
            [self.model.joint(name).dofadr[0] for name in MUJOCO_DOF_ORDER]
        )

    def run(
        self,
        run: RunData,
        *,
        delay_sec: float,
        target_kp: float,
        target_kd: float,
    ) -> ReplayResult:
        if delay_sec < 0.0:
            raise ValueError("delay_sec은 0 이상이어야 합니다.")
        command_dt = 1.0 / run.command_rate_hz
        ratio = command_dt / self.config.physics_dt_sec
        substeps = int(round(ratio))
        if not np.isclose(ratio, substeps, rtol=0.0, atol=1e-9):
            raise ValueError("command 주기는 physics_dt의 정수배여야 합니다.")

        median_rx_delay = float(
            np.median(run.rx_time_ns - run.tx_time_ns) * 1e-9
        )
        sample_substep = int(
            np.clip(round(median_rx_delay / self.config.physics_dt_sec), 1, substeps)
        )

        data = mujoco.MjData(self.model)
        data.qpos[self.qpos_indices] = run.q_init_rad
        data.qvel[self.dof_indices] = run.dq_init_rad_s
        mujoco.mj_forward(self.model, data)

        kp = self.config.kp_baseline.copy()
        kd = self.config.kd_baseline.copy()
        kp[run.target_index] = target_kp
        kd[run.target_index] = target_kd
        sample_count = len(run.q_cmd_rad)
        q = np.empty((sample_count, len(MUJOCO_DOF_ORDER)))
        dq = np.empty_like(q)
        tau = np.empty_like(q)

        for cycle in range(sample_count):
            recorded = False
            for substep in range(substeps):
                sim_time = cycle * command_dt + substep * self.config.physics_dt_sec
                delayed_time = sim_time - delay_sec
                command_index = int(np.floor(delayed_time / command_dt + 1e-12))
                command_index = min(max(command_index, 0), sample_count - 1)
                q_now = data.qpos[self.qpos_indices]
                dq_now = data.qvel[self.dof_indices]
                torque = kp * (run.q_cmd_rad[command_index] - q_now) - kd * dq_now
                torque = np.clip(
                    torque,
                    -self.config.torque_limit_nm,
                    self.config.torque_limit_nm,
                )
                data.qfrc_applied[self.dof_indices] = torque
                mujoco.mj_step(self.model, data)
                if substep + 1 == sample_substep:
                    q[cycle] = data.qpos[self.qpos_indices]
                    dq[cycle] = data.qvel[self.dof_indices]
                    tau[cycle] = torque
                    recorded = True
            if not recorded:
                raise RuntimeError("simulation sample을 기록하지 못했습니다.")
        return ReplayResult(q, dq, tau, sample_substep)
