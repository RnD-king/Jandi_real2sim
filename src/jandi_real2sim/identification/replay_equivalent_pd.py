from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from jandi_real2sim.config import MUJOCO_DOF_ORDER
from jandi_real2sim.identification.dataset import RunData


@dataclass(frozen=True)
class EquivalentReplayConfig:
    model_xml: Path
    physics_dt_sec: float
    torque_limit_nm: float


@dataclass(frozen=True)
class EquivalentParameters:
    kp_eff: float
    kd_eff: float
    backlash_total_rad: float
    coulomb_friction_nm: float


@dataclass(frozen=True)
class EquivalentReplayResult:
    q_rad: np.ndarray
    dq_rad_s: np.ndarray
    tau_nm: np.ndarray
    effective_error_rad: np.ndarray
    sample_substep: int


@dataclass(frozen=True)
class JointwiseEquivalentParameters:
    """12관절별 PD와 공통 상태형 백래시 파라미터."""

    kp_eff: np.ndarray
    kd_eff: np.ndarray
    backlash_total_rad: float
    coulomb_friction_nm: float
    position_quantization_rad: float


def stateful_play_step(
    previous: np.ndarray,
    input_position: np.ndarray,
    total_width_rad: float,
) -> np.ndarray:
    """Jandi 학습 actuator와 같은 symmetric play operator 한 step."""
    if total_width_rad < 0.0:
        raise ValueError("backlash_total_rad는 0 이상이어야 합니다.")
    half_width = 0.5 * total_width_rad
    return np.maximum(
        np.minimum(previous, input_position + half_width),
        input_position - half_width,
    )


def _quantize_position(value: np.ndarray, quantum_rad: float) -> np.ndarray:
    if quantum_rad < 0.0:
        raise ValueError("position_quantization_rad는 0 이상이어야 합니다.")
    if quantum_rad == 0.0:
        return value
    return np.round(value / quantum_rad) * quantum_rad


def backlash_deadband(error_rad: np.ndarray, total_width_rad: float) -> np.ndarray:
    """전체 폭 ``b``인 대칭 위치 데드밴드: |e| <= b/2이면 토크 오차 0."""
    if total_width_rad < 0.0:
        raise ValueError("backlash_total_rad는 0 이상이어야 합니다.")
    half_width = 0.5 * total_width_rad
    return np.sign(error_rad) * np.maximum(np.abs(error_rad) - half_width, 0.0)


class FixedBaseEquivalentPdReplay:
    """고정 베이스 Jandi의 등가 PD + 위치 데드밴드 + Coulomb 마찰 모델."""

    def __init__(self, config: EquivalentReplayConfig):
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
        if self.model.nq != 12 or self.model.nv != 12 or self.model.nu != 0:
            raise ValueError(
                f"식별 모델 자유도 불일치: nq={self.model.nq}, "
                f"nv={self.model.nv}, nu={self.model.nu}"
            )
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
        parameters: EquivalentParameters,
    ) -> EquivalentReplayResult:
        if delay_sec < 0.0:
            raise ValueError("delay_sec은 0 이상이어야 합니다.")
        command_dt = 1.0 / run.command_rate_hz
        ratio = command_dt / self.config.physics_dt_sec
        substeps = int(round(ratio))
        if not np.isclose(ratio, substeps, rtol=0.0, atol=1e-9):
            raise ValueError("command 주기는 physics_dt의 정수배여야 합니다.")

        median_rx_delay = float(np.median(run.rx_time_ns - run.tx_time_ns) * 1e-9)
        sample_substep = int(
            np.clip(round(median_rx_delay / self.config.physics_dt_sec), 1, substeps)
        )
        # MuJoCo frictionloss는 dof별 Coulomb 마찰 한계 토크이다.
        self.model.dof_frictionloss[self.dof_indices] = parameters.coulomb_friction_nm

        data = mujoco.MjData(self.model)
        data.qpos[self.qpos_indices] = run.q_init_rad
        data.qvel[self.dof_indices] = run.dq_init_rad_s
        mujoco.mj_forward(self.model, data)

        sample_count = len(run.q_cmd_rad)
        q = np.empty((sample_count, 12), dtype=np.float64)
        dq = np.empty_like(q)
        tau = np.empty_like(q)
        effective_error = np.empty_like(q)

        for cycle in range(sample_count):
            recorded = False
            for substep in range(substeps):
                sim_time = cycle * command_dt + substep * self.config.physics_dt_sec
                delayed_time = sim_time - delay_sec
                command_index = int(np.floor(delayed_time / command_dt + 1e-12))
                command_index = min(max(command_index, 0), sample_count - 1)
                q_now = data.qpos[self.qpos_indices]
                dq_now = data.qvel[self.dof_indices]
                error = backlash_deadband(
                    run.q_cmd_rad[command_index] - q_now,
                    parameters.backlash_total_rad,
                )
                torque = parameters.kp_eff * error - parameters.kd_eff * dq_now
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
                    effective_error[cycle] = error
                    recorded = True
            if not recorded:
                raise RuntimeError("simulation sample을 기록하지 못했습니다.")
        return EquivalentReplayResult(q, dq, tau, effective_error, sample_substep)


class FixedBaseStatefulJointwisePdReplay(FixedBaseEquivalentPdReplay):
    """배포용 actuator와 같은 tick + stateful play + 관절별 PD replay."""

    def run(
        self,
        run: RunData,
        *,
        delay_sec: float,
        parameters: JointwiseEquivalentParameters,
    ) -> EquivalentReplayResult:
        if delay_sec < 0.0:
            raise ValueError("delay_sec은 0 이상이어야 합니다.")
        kp = np.asarray(parameters.kp_eff, dtype=np.float64)
        kd = np.asarray(parameters.kd_eff, dtype=np.float64)
        if kp.shape != (12,) or kd.shape != (12,):
            raise ValueError("kp_eff/kd_eff는 MuJoCo 순서의 12개 값이어야 합니다.")
        if np.any(kp < 0.0) or np.any(kd < 0.0):
            raise ValueError("PD gain은 0 이상이어야 합니다.")

        command_dt = 1.0 / run.command_rate_hz
        ratio = command_dt / self.config.physics_dt_sec
        substeps = int(round(ratio))
        if not np.isclose(ratio, substeps, rtol=0.0, atol=1e-9):
            raise ValueError("command 주기는 physics_dt의 정수배여야 합니다.")
        median_rx_delay = float(
            np.median(run.rx_time_ns - run.tx_time_ns) * 1e-9
        )
        sample_substep = int(
            np.clip(
                round(median_rx_delay / self.config.physics_dt_sec),
                1,
                substeps,
            )
        )

        self.model.dof_frictionloss[self.dof_indices] = (
            parameters.coulomb_friction_nm
        )
        data = mujoco.MjData(self.model)
        data.qpos[self.qpos_indices] = run.q_init_rad
        data.qvel[self.dof_indices] = run.dq_init_rad_s
        mujoco.mj_forward(self.model, data)

        sample_count = len(run.q_cmd_rad)
        q = np.empty((sample_count, 12), dtype=np.float64)
        dq = np.empty_like(q)
        tau = np.empty_like(q)
        effective_error = np.empty_like(q)
        transmitted_target = _quantize_position(
            run.q_cmd_rad[0].copy(),
            parameters.position_quantization_rad,
        )

        for cycle in range(sample_count):
            recorded = False
            for substep in range(substeps):
                sim_time = (
                    cycle * command_dt
                    + substep * self.config.physics_dt_sec
                )
                delayed_time = sim_time - delay_sec
                command_index = int(
                    np.floor(delayed_time / command_dt + 1e-12)
                )
                command_index = min(max(command_index, 0), sample_count - 1)
                target = _quantize_position(
                    run.q_cmd_rad[command_index],
                    parameters.position_quantization_rad,
                )
                transmitted_target = stateful_play_step(
                    transmitted_target,
                    target,
                    parameters.backlash_total_rad,
                )
                q_now = data.qpos[self.qpos_indices]
                dq_now = data.qvel[self.dof_indices]
                measured = _quantize_position(
                    q_now,
                    parameters.position_quantization_rad,
                )
                error = transmitted_target - measured
                torque = kp * error - kd * dq_now
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
                    effective_error[cycle] = error
                    recorded = True
            if not recorded:
                raise RuntimeError("simulation sample을 기록하지 못했습니다.")
        return EquivalentReplayResult(
            q,
            dq,
            tau,
            effective_error,
            sample_substep,
        )
