from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .analysis import Run, _state
from .config import Bench, Mode5Campaign


@dataclass(frozen=True)
class Replay:
    time_s: np.ndarray
    q_real: np.ndarray
    dq_real: np.ndarray
    current_real: np.ndarray
    q_sim: np.ndarray
    dq_sim: np.ndarray
    current_model: np.ndarray
    q_cmd_delayed: np.ndarray


def _bench_mass_properties(bench: Bench) -> tuple[float, float, float]:
    if bench.bare_horn:
        # 회전자·기어·혼 관성은 fitted armature에 포함한다. 아래 body inertia는
        # MuJoCo 모델 생성을 위한 수치적으로 무시 가능한 양의 값이다.
        return 1.0e-9, 1.0e-6, 1.0e-12
    assert bench.arm_mass_kg is not None
    assert bench.arm_com_radius_m is not None
    assert bench.arm_inertia_kg_m2 is not None
    load_mass = bench.added_load_mass_kg or 0.0
    load_radius = bench.added_load_radius_m or 0.0
    total_mass = bench.arm_mass_kg + load_mass
    com_radius = (
        bench.arm_mass_kg * bench.arm_com_radius_m + load_mass * load_radius
    ) / total_mass
    pivot_inertia = bench.arm_inertia_kg_m2 + load_mass * load_radius**2
    com_inertia = pivot_inertia - total_mass * com_radius**2
    if total_mass <= 0 or com_radius < 0 or com_inertia <= 0:
        raise ValueError(
            "시험대 mass/COM/inertia가 물리적으로 유효하지 않습니다: "
            f"mass={total_mass}, com={com_radius}, I_com={com_inertia}"
        )
    return total_mass, com_radius, com_inertia


def build_model(bench: Bench, params: dict[str, float], dt: float = 0.002) -> mujoco.MjModel:
    mass, com_radius, com_inertia = _bench_mass_properties(bench)
    assert bench.gravity_zero_offset_rad is not None
    armature = float(params["armature_kg_m2"])
    friction = float(params["frictionloss_Nm"])
    damping = float(params["damping_Nm_s_per_rad"])
    if min(armature, friction, damping) < 0:
        raise ValueError("MuJoCo armature/frictionloss/damping은 음수일 수 없습니다.")
    half = bench.gravity_zero_offset_rad / 2.0
    quat = f"{math.cos(half):.12g} 0 {math.sin(half):.12g} 0"
    xml = f"""
<mujoco model="mx106_mode5_bench">
  <option timestep="{dt:.12g}" gravity="0 0 -9.80665" integrator="implicitfast"/>
  <worldbody>
    <body name="bench_arm" quat="{quat}">
      <joint name="output_joint" type="hinge" axis="0 1 0"
             armature="{armature:.12g}" frictionloss="{friction:.12g}"
             damping="{damping:.12g}" limited="false"/>
      <inertial pos="0 0 {-com_radius:.12g}" mass="{mass:.12g}"
                diaginertia="{com_inertia:.12g} {com_inertia:.12g} {com_inertia:.12g}"/>
      <geom type="capsule" fromto="0 0 0 0 0 {-2 * com_radius:.12g}"
            size="0.005" mass="0" contype="0" conaffinity="0" rgba="0.4 0.4 0.4 1"/>
    </body>
  </worldbody>
  <actuator><motor name="joint_torque" joint="output_joint" gear="1"/></actuator>
</mujoco>
"""
    return mujoco.MjModel.from_xml_string(xml)


def replay(cfg: Mode5Campaign, run: Run, params: dict[str, float]) -> Replay:
    state = _state(run)
    time_s = state["time_s"]
    time_s = time_s - time_s[0]
    q_real = state["q_present_rad"]
    dq_real = state["dq_present_rad_s"]
    current_real = state["current_A_joint"]
    events_t = (run.events["tx_end_ns"] - run.events["tx_end_ns"][0]) * 1e-9
    events_goal = run.events["goal_rad"]

    model = build_model(cfg.benches[run.condition], params)
    data = mujoco.MjData(model)
    data.qpos[0] = q_real[0]
    data.qvel[0] = dq_real[0]
    mujoco.mj_forward(model, data)

    delay = float(params["delay_s"])
    a_p = float(params["aP_A_per_rad"])
    a_d = float(params["aD_A_s_per_rad"])
    k_tau = float(params["Ktau_Nm_per_A"])
    current_cap = float(params["current_cap_A"])
    if a_p <= 0 or k_tau <= 0 or current_cap <= 0 or delay < 0 or a_d < 0:
        raise ValueError("Mode 5 controller 파라미터의 부호/범위가 유효하지 않습니다.")

    sim_time: list[float] = []
    sim_q: list[float] = []
    sim_dq: list[float] = []
    sim_current: list[float] = []
    sim_goal: list[float] = []
    event_index = 0
    end_time = float(time_s[-1])
    while data.time <= end_time + model.opt.timestep:
        delayed_time = max(0.0, data.time - delay)
        while event_index + 1 < len(events_t) and events_t[event_index + 1] <= delayed_time:
            event_index += 1
        goal = float(events_goal[event_index])
        current = float(np.clip(a_p * (goal - data.qpos[0]) - a_d * data.qvel[0], -current_cap, current_cap))
        data.ctrl[0] = k_tau * current
        sim_time.append(data.time)
        sim_q.append(float(data.qpos[0]))
        sim_dq.append(float(data.qvel[0]))
        sim_current.append(current)
        sim_goal.append(goal)
        mujoco.mj_step(model, data)

    t = np.asarray(sim_time)
    return Replay(
        time_s=time_s,
        q_real=q_real,
        dq_real=dq_real,
        current_real=current_real,
        q_sim=np.interp(time_s, t, np.asarray(sim_q)),
        dq_sim=np.interp(time_s, t, np.asarray(sim_dq)),
        current_model=np.interp(time_s, t, np.asarray(sim_current)),
        q_cmd_delayed=np.interp(time_s, t, np.asarray(sim_goal)),
    )
