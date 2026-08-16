from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from jandi_real2sim.config import MUJOCO_DOF_ORDER, RobotConfig
from jandi_real2sim.identification.dataset import RunData, interpolate_state


@dataclass(frozen=True)
class PwmReplayConfig:
    model_xml: Path
    robot: RobotConfig
    physics_dt_sec: float
    nominal_voltage_v: float
    torque_limit_nm: float


@dataclass(frozen=True)
class M1Parameters:
    drive_gain_nm_per_duty: float
    armature_kg_m2: float
    coulomb_friction_nm: float
    viscous_friction_nm_s_per_rad: float


@dataclass(frozen=True)
class PwmReplayResult:
    q_rad: np.ndarray
    dq_rad_s: np.ndarray
    drive_torque_nm: np.ndarray
    pwm_duty: np.ndarray
    sample_substep: int


class FixedBasePwmReplay:
    """실측 Present PWM을 입력으로 고정-base Jandi M1을 재생한다.

    M1은 출력축 기준 등가모델이다. drive gain에는 모터 상수, 기어비와
    전달 효율이 함께 들어가고, Coulomb/viscous/armature는 MuJoCo joint
    파라미터로 적용한다. 내부 Position P/D나 command delay는 여기서 다시
    피팅하지 않는다.
    """

    def __init__(self, config: PwmReplayConfig):
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
                "식별 모델은 fixed base 12-DoF/no-actuator여야 합니다: "
                f"nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}"
            )
        self.qpos_indices = np.asarray(
            [self.model.joint(name).qposadr[0] for name in MUJOCO_DOF_ORDER]
        )
        self.dof_indices = np.asarray(
            [self.model.joint(name).dofadr[0] for name in MUJOCO_DOF_ORDER]
        )
        by_name = config.robot.by_name
        self.directions = np.asarray(
            [float(by_name[name].direction) for name in MUJOCO_DOF_ORDER]
        )
        self.motor_ids = tuple(int(by_name[name].motor_id) for name in MUJOCO_DOF_ORDER)

    def _pwm_limit_raw(self, run: RunData) -> np.ndarray:
        settings = run.metadata.get("actuator_settings", {})
        values = []
        for motor_id in self.motor_ids:
            item = settings.get(str(motor_id), settings.get(motor_id))
            if item is None or "pwm_limit_raw" not in item:
                raise ValueError(
                    f"{run.run_dir}: motor ID {motor_id}의 pwm_limit_raw가 없습니다."
                )
            limit = float(item["pwm_limit_raw"])
            if limit <= 0.0:
                raise ValueError(f"{run.run_dir}: 잘못된 PWM limit: {limit}")
            values.append(limit)
        return np.asarray(values)

    @staticmethod
    def _interpolate_matrix(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if np.count_nonzero(mask) < 2:
            raise ValueError("PWM replay에는 state 표본이 최소 2개 필요합니다.")
        return np.column_stack(
            [interpolate_state(values[:, index], mask) for index in range(values.shape[1])]
        )

    def run(self, run: RunData, parameters: M1Parameters) -> PwmReplayResult:
        if min(
            parameters.drive_gain_nm_per_duty,
            parameters.armature_kg_m2,
            parameters.coulomb_friction_nm,
            parameters.viscous_friction_nm_s_per_rad,
        ) < 0.0:
            raise ValueError("M1 파라미터는 음수일 수 없습니다.")

        command_dt = 1.0 / run.command_rate_hz
        ratio = command_dt / self.config.physics_dt_sec
        substeps = int(round(ratio))
        if not np.isclose(ratio, substeps, rtol=0.0, atol=1e-9):
            raise ValueError("command 주기는 physics_dt의 정수배여야 합니다.")
        median_rx_delay = float(np.median(run.rx_time_ns - run.tx_time_ns) * 1e-9)
        sample_substep = int(
            np.clip(round(median_rx_delay / self.config.physics_dt_sec), 1, substeps)
        )

        pwm_raw = self._interpolate_matrix(run.present_pwm_raw, run.state_mask)
        voltage = self._interpolate_matrix(run.input_voltage_v, run.state_mask)
        duty = pwm_raw / self._pwm_limit_raw(run)[None, :]
        duty *= self.directions[None, :]
        duty = np.clip(duty, -1.0, 1.0)

        # 모든 하체 관절이 같은 MX-106을 사용하므로 우선 공통 M1을 적용한다.
        self.model.dof_armature[self.dof_indices] = parameters.armature_kg_m2
        self.model.dof_frictionloss[self.dof_indices] = parameters.coulomb_friction_nm
        self.model.dof_damping[self.dof_indices] = parameters.viscous_friction_nm_s_per_rad

        data = mujoco.MjData(self.model)
        data.qpos[self.qpos_indices] = run.q_init_rad
        data.qvel[self.dof_indices] = run.dq_init_rad_s
        mujoco.mj_setConst(self.model, data)
        mujoco.mj_forward(self.model, data)

        sample_count = len(run.q_cmd_rad)
        q = np.empty((sample_count, 12), dtype=np.float64)
        dq = np.empty_like(q)
        torque_log = np.empty_like(q)
        duty_log = np.empty_like(q)

        for cycle in range(sample_count):
            recorded = False
            for substep in range(substeps):
                # State block에서 관측된 PWM은 해당 cycle의 sample 시점부터
                # 유효하다고 두고, 그 전 substep에는 직전 관측값을 hold한다.
                input_cycle = cycle if substep + 1 >= sample_substep else max(cycle - 1, 0)
                voltage_scale = voltage[input_cycle] / self.config.nominal_voltage_v
                drive_torque = (
                    parameters.drive_gain_nm_per_duty
                    * duty[input_cycle]
                    * voltage_scale
                )
                drive_torque = np.clip(
                    drive_torque,
                    -self.config.torque_limit_nm,
                    self.config.torque_limit_nm,
                )
                data.qfrc_applied[self.dof_indices] = drive_torque
                mujoco.mj_step(self.model, data)
                if substep + 1 == sample_substep:
                    q[cycle] = data.qpos[self.qpos_indices]
                    dq[cycle] = data.qvel[self.dof_indices]
                    torque_log[cycle] = drive_torque
                    duty_log[cycle] = duty[input_cycle]
                    recorded = True
            if not recorded:
                raise RuntimeError("simulation sample을 기록하지 못했습니다.")
        return PwmReplayResult(q, dq, torque_log, duty_log, sample_substep)
