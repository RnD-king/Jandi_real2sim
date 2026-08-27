"""Standalone one-DOF MuJoCo replay with direct current-domain torque input."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .canonical_acquisition import CURRENT_A_PER_RAW
from .canonical_config import CanonicalCampaign
from .spec import CANONICAL_HINGE_AXIS


@dataclass(frozen=True)
class Replay:
    time_sec: np.ndarray
    q_rad: np.ndarray
    qd_rad_s: np.ndarray
    current_A: np.ndarray


def zoh_command(query_time_sec: np.ndarray, command_time_sec: np.ndarray,
                command_goal_rad: np.ndarray) -> np.ndarray:
    """Reconstruct the latest actually transmitted command (previous-value hold)."""
    query = np.asarray(query_time_sec, dtype=float)
    event_time = np.asarray(command_time_sec, dtype=float)
    goals = np.asarray(command_goal_rad, dtype=float)
    if event_time.ndim != 1 or goals.ndim != 1 or len(event_time) != len(goals) or not len(goals):
        raise ValueError("ZOH command time/goal은 같은 길이의 비어 있지 않은 1-D 배열이어야 합니다.")
    if np.any(np.diff(event_time) < 0):
        raise ValueError("command timestamp는 nondecreasing이어야 합니다.")
    indices = np.searchsorted(event_time, query, side="right") - 1
    indices = np.clip(indices, 0, len(goals) - 1)
    return goals[indices]


def _resolved(cfg: CanonicalCampaign, mechanical: str, resolved: dict | None) -> tuple[dict, dict, dict, object]:
    if resolved is None:
        item = cfg.configuration(mechanical)
        return cfg.geometry, cfg.loads, cfg.registers, item
    geometry, loads, registers = resolved["geometry"], resolved["loads"], resolved["controller"]
    item = cfg.configuration(mechanical)
    return geometry, loads, registers, item


def _mass_properties(cfg: CanonicalCampaign, mechanical: str, resolved: dict | None = None) -> tuple[float, float, float]:
    geometry, loads, _registers, item = _resolved(cfg, mechanical, resolved)
    arm_mass = float(geometry["arm_mass_kg"])
    arm_com = float(geometry["arm_com_radius_m"])
    arm_inertia = float(geometry["arm_inertia_kg_m2"])
    load_mass = float(loads[item.load]["measured_mass_kg"])
    load_radius = float(geometry["arm_lengths_m"][item.arm_length])
    total_mass = arm_mass + load_mass
    com_radius = (arm_mass * arm_com + load_mass * load_radius) / total_mass
    if geometry["arm_inertia_reference"] == "about_com":
        pivot_inertia = arm_inertia + arm_mass * arm_com**2 + load_mass * load_radius**2
    else:
        pivot_inertia = arm_inertia + load_mass * load_radius**2
    com_inertia = pivot_inertia - total_mass * com_radius**2
    if min(total_mass, com_radius, pivot_inertia, com_inertia) <= 0:
        raise ValueError(
            "bench mass/COM/inertia 조합이 물리적으로 유효하지 않습니다: "
            f"mass={total_mass}, com={com_radius}, I_pivot={pivot_inertia}, I_com={com_inertia}"
        )
    return total_mass, com_radius, com_inertia


def build_model(cfg: CanonicalCampaign, mechanical: str, params: dict[str, float], dt: float,
                resolved: dict | None = None) -> mujoco.MjModel:
    if dt <= 0:
        raise ValueError("MuJoCo physics timestep은 양수여야 합니다.")
    for name in ("armature_kg_m2", "coulomb_friction_Nm", "viscous_friction_Nm_s_per_rad"):
        if float(params[name]) < 0:
            raise ValueError(f"{name}은 음수일 수 없습니다.")
    geometry, _loads, _registers, _item = _resolved(cfg, mechanical, resolved)
    mass, radius, inertia_com = _mass_properties(cfg, mechanical, resolved)
    gravity = float(geometry["gravity_m_s2"])
    offset = float(geometry["gravity_zero_angle_rad"])
    gravity_sign = float(geometry["gravity_torque_sign"])
    x = -gravity_sign * radius * math.sin(offset)
    z = -gravity_sign * radius * math.cos(offset)
    inertia = max(inertia_com, 1e-12)
    xml = f"""
<mujoco model="mx106_mode5_current_domain_m1">
  <option timestep="{dt:.12g}" gravity="0 0 {-gravity:.12g}" integrator="implicitfast"/>
  <worldbody>
    <body name="pendulum">
      <joint name="output" type="hinge" axis="{CANONICAL_HINGE_AXIS[0]:g} {CANONICAL_HINGE_AXIS[1]:g} {CANONICAL_HINGE_AXIS[2]:g}"
             armature="{float(params['armature_kg_m2']):.12g}"
             frictionloss="{float(params['coulomb_friction_Nm']):.12g}"
             damping="{float(params['viscous_friction_Nm_s_per_rad']):.12g}"/>
      <inertial pos="{x:.12g} 0 {z:.12g}" mass="{mass:.12g}"
                diaginertia="{inertia:.12g} {inertia:.12g} {inertia:.12g}"/>
    </body>
  </worldbody>
  <actuator><motor name="direct_torque" joint="output" gear="1"/></actuator>
</mujoco>
"""
    return mujoco.MjModel.from_xml_string(xml)


def replay(
    cfg: CanonicalCampaign,
    mechanical: str,
    measured_time_sec: np.ndarray,
    command_time_sec: np.ndarray,
    command_goal_rad: np.ndarray,
    q0: float,
    qd0: float,
    params: dict[str, float],
    dt: float,
    resolved: dict | None = None,
) -> Replay:
    model = build_model(cfg, mechanical, params, dt, resolved)
    data = mujoco.MjData(model)
    data.qpos[0], data.qvel[0] = q0, qd0
    mujoco.mj_forward(model, data)
    delay = float(params["delay_s"])
    a_p = float(params["aP_A_per_rad"])
    a_d = float(params["aD_A_s_per_rad"])
    k_tau = float(params["Ktau_eff_Nm_per_A"])
    if delay < 0 or a_p <= 0 or a_d < 0 or k_tau <= 0:
        raise ValueError("delay/aP/aD/Ktau의 물리 부호가 유효하지 않습니다.")
    _geometry, _loads, registers, _item = _resolved(cfg, mechanical, resolved)
    raw_cap = min(abs(int(registers["goal_current_raw"])), int(registers["expected_current_limit_raw"]))
    current_cap = raw_cap * CURRENT_A_PER_RAW
    sim_t: list[float] = []
    sim_q: list[float] = []
    sim_qd: list[float] = []
    sim_i: list[float] = []
    end = float(measured_time_sec[-1])
    goal_trace = zoh_command(
        np.arange(0.0, end + 2.0 * dt, dt) - delay,
        command_time_sec,
        command_goal_rad,
    )
    goal_index = 0
    while data.time <= end + dt:
        goal = float(goal_trace[min(goal_index, len(goal_trace) - 1)])
        current = float(np.clip(a_p * (goal - data.qpos[0]) - a_d * data.qvel[0], -current_cap, current_cap))
        data.ctrl[0] = k_tau * current
        sim_t.append(data.time)
        sim_q.append(float(data.qpos[0]))
        sim_qd.append(float(data.qvel[0]))
        sim_i.append(current)
        mujoco.mj_step(model, data)
        goal_index += 1
    t = np.asarray(sim_t)
    return Replay(
        time_sec=measured_time_sec,
        q_rad=np.interp(measured_time_sec, t, np.asarray(sim_q)),
        qd_rad_s=np.interp(measured_time_sec, t, np.asarray(sim_qd)),
        current_A=np.interp(measured_time_sec, t, np.asarray(sim_i)),
    )
