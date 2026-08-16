from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

import numpy as np

from jandi_real2sim.config import MUJOCO_DOF_ORDER
from jandi_real2sim.identification.fit_m0 import (
    Candidate,
    _huber,
    _shortlist_delay_values,
    load_fit_config,
)
from jandi_real2sim.identification.dataset import RunData
from jandi_real2sim.identification.fit_m0_dual_gain import (
    dynamic_replay_loss,
    load_dual_gain_fit_config,
)
from jandi_real2sim.identification.fit_m1_pwm import load_pwm_m1_fit_config
from jandi_real2sim.identification.fit_m1_staged import (
    _refinement_bounds,
    load_staged_m1_fit_config,
)
from jandi_real2sim.identification.fit_equivalent_pd import (
    equivalent_loss_parts,
    load_equivalent_fit_config,
)
from jandi_real2sim.identification.fit_joint34_pd import (
    _parameters as joint34_parameters,
    load_joint34_fit_config,
)
from jandi_real2sim.identification.replay import FixedBaseReplay, ReplayResult
from jandi_real2sim.identification.replay_equivalent_pd import (
    EquivalentReplayResult,
    FixedBaseEquivalentPdReplay,
    FixedBaseStatefulJointwisePdReplay,
    backlash_deadband,
    stateful_play_step,
)
from jandi_real2sim.identification.replay_pwm import (
    FixedBasePwmReplay,
    M1Parameters,
)


class IdentificationStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).parents[1]

    def test_huber_is_quadratic_then_linear(self) -> None:
        actual = _huber(np.asarray([0.05, 0.20]), delta=0.10)
        np.testing.assert_allclose(actual, [0.00125, 0.015])

    def test_fit_config_and_fixed_base_model(self) -> None:
        config = load_fit_config(self.root / "configs" / "m0_ankle_roll.yaml")
        replay = FixedBaseReplay(config.replay)
        self.assertEqual(replay.model.nq, 12)
        self.assertEqual(replay.model.nv, 12)
        self.assertEqual(replay.model.nu, 0)
        self.assertAlmostEqual(replay.model.opt.timestep, 0.002)
        self.assertEqual(config.target_joints, ("RL6_joint", "LL6_joint"))
        self.assertEqual(len(config.delay_values_sec), 26)

    def test_delay_search_only_refines_best_neighborhoods(self) -> None:
        delays = np.arange(0.0, 0.012, 0.002)
        losses = [9.0, 1.0, 2.0, 8.0, 0.5, 7.0]
        screened = tuple(
            Candidate(float(delay), 8.0, 0.7, loss, True, 1)
            for delay, loss in zip(delays, losses)
        )
        selected = _shortlist_delay_values(screened, delays, 2, 1)
        self.assertEqual(selected, (0.0, 0.002, 0.004, 0.006, 0.008, 0.010))

    def test_dual_gain_config_uses_shared_zero_to_ten_ms_delay_grid(self) -> None:
        config = load_dual_gain_fit_config(self.root / "configs" / "m0_dual_gain.yaml")
        np.testing.assert_allclose(
            config.delay_values_sec, np.arange(0.0, 0.012, 0.002)
        )
        self.assertEqual(
            [(item.name, item.register_p) for item in config.conditions],
            [("P350", 350), ("P850", 850)],
        )

    def test_dynamic_loss_ignores_direction_dependent_plateau_offset(self) -> None:
        config = load_dual_gain_fit_config(self.root / "configs" / "m0_dual_gain.yaml")
        config = dataclasses.replace(
            config,
            transient_window_sec=0.20,
            plateau_tail_sec=0.10,
            velocity_loss_weight=0.0,
        )
        count = 100
        q_cmd = np.zeros((count, 12))
        q_cmd[20:70, 5] = 0.1
        q_real = np.zeros_like(q_cmd)
        q_real[20:70, 5] = 0.09
        q_real[70:, 5] = 0.01
        q_sim = q_real.copy()
        q_sim[20:70, 5] += 0.01
        q_sim[70:, 5] -= 0.01
        zeros = np.zeros_like(q_cmd)
        run = RunData(
            run_dir=Path("synthetic"),
            metadata={"command_rate_hz": 100},
            target_joint="RL6_joint",
            repeat_index=1,
            split_role="fit",
            phase=tuple("synthetic" for _ in range(count)),
            q_cmd_rad=q_cmd,
            q_real_rad=q_real,
            dq_real_rad_s=zeros,
            q_trajectory_rad=q_cmd.copy(),
            dq_trajectory_rad_s=zeros,
            present_pwm_raw=zeros,
            present_current_a=zeros,
            input_voltage_v=np.full_like(q_cmd, 12.0),
            state_mask=np.ones(count, dtype=bool),
            overrun_mask=np.zeros(count, dtype=bool),
            tx_time_ns=np.arange(count, dtype=np.int64),
            rx_time_ns=np.arange(count, dtype=np.int64),
            q_init_rad=np.zeros(12),
            dq_init_rad_s=np.zeros(12),
        )
        result = ReplayResult(q_sim, zeros, zeros, sample_substep=1)
        self.assertAlmostEqual(dynamic_replay_loss(run, result, config), 0.0)

    def test_pwm_m1_config_and_replay_are_static_and_finite(self) -> None:
        config = load_pwm_m1_fit_config(self.root / "configs" / "m1_pwm.yaml")
        replay = FixedBasePwmReplay(config.replay)
        self.assertEqual(replay.model.nq, 12)
        self.assertEqual(replay.model.nu, 0)
        self.assertEqual(len(config.initial_starts), 3)

        count = 20
        pose = np.asarray(
            [config.replay.robot.by_name[name].walking_rad for name in config.replay.robot.by_name]
        )
        q_cmd = np.repeat(pose[None, :], count, axis=0)
        zeros = np.zeros_like(q_cmd)
        settings = {
            str(config.replay.robot.by_name[name].motor_id): {"pwm_limit_raw": 885}
            for name in config.replay.robot.by_name
        }
        run = RunData(
            run_dir=Path("synthetic_pwm"),
            metadata={"command_rate_hz": 100, "actuator_settings": settings},
            target_joint="RL6_joint",
            repeat_index=1,
            split_role="fit",
            phase=tuple("synthetic" for _ in range(count)),
            q_cmd_rad=q_cmd,
            q_real_rad=q_cmd.copy(),
            dq_real_rad_s=zeros,
            q_trajectory_rad=q_cmd.copy(),
            dq_trajectory_rad_s=zeros,
            present_pwm_raw=zeros,
            present_current_a=zeros,
            input_voltage_v=np.full_like(q_cmd, 12.0),
            state_mask=np.ones(count, dtype=bool),
            overrun_mask=np.zeros(count, dtype=bool),
            tx_time_ns=np.arange(count, dtype=np.int64) * 10_000_000,
            rx_time_ns=np.arange(count, dtype=np.int64) * 10_000_000 + 2_000_000,
            q_init_rad=pose,
            dq_init_rad_s=np.zeros(12),
        )
        result = replay.run(run, config.initial_starts[0])
        self.assertTrue(np.all(np.isfinite(result.q_rad)))
        np.testing.assert_allclose(result.pwm_duty, 0.0)
        np.testing.assert_allclose(result.drive_torque_nm, 0.0)

    def test_staged_pwm_m1_config_has_three_bounded_stages(self) -> None:
        config = load_staged_m1_fit_config(
            self.root / "configs" / "m1_pwm_staged.yaml"
        )
        self.assertEqual(config.base.target_joints, ("RL6_joint", "LL6_joint"))
        self.assertEqual(len(config.triangle.starts), 2)
        self.assertEqual(len(config.multisine.starts), 2)
        self.assertEqual(len(config.final.starts), 2)
        center = M1Parameters(4.0, 0.02, 0.1, 0.2)
        bounds = _refinement_bounds(center, config)
        for value, bound, global_bound in zip(
            (4.0, 0.02, 0.1, 0.2), bounds, config.base.bounds
        ):
            self.assertLessEqual(bound[0], value)
            self.assertGreaterEqual(bound[1], value)
            self.assertGreaterEqual(bound[0], global_bound[0])
            self.assertLessEqual(bound[1], global_bound[1])

    def test_equivalent_pd_config_model_and_deadband(self) -> None:
        config = load_equivalent_fit_config(
            self.root / "configs" / "equivalent_pd.yaml"
        )
        replay = FixedBaseEquivalentPdReplay(config.replay)
        self.assertEqual(replay.model.nq, 12)
        self.assertEqual(replay.model.nu, 0)
        np.testing.assert_allclose(
            backlash_deadband(np.asarray([-0.02, -0.005, 0.0, 0.005, 0.02]), 0.02),
            [-0.01, 0.0, 0.0, 0.0, 0.01],
        )
        self.assertEqual(len(config.delay_values_sec), 11)
        self.assertAlmostEqual(
            config.conditions[1].initial_kp / config.conditions[0].initial_kp,
            850.0 / 350.0,
        )
        self.assertEqual(
            config.conditions[0].initial_kd,
            config.conditions[1].initial_kd,
        )
        self.assertAlmostEqual(config.fixed_kd_eff, 0.60)

    def test_equivalent_loss_keeps_absolute_plateau_error(self) -> None:
        config = load_equivalent_fit_config(
            self.root / "configs" / "equivalent_pd.yaml"
        )
        config = dataclasses.replace(
            config,
            transient_window_sec=0.20,
            plateau_tail_sec=0.10,
            velocity_loss_weight=0.0,
        )
        count = 100
        q_cmd = np.zeros((count, 12))
        q_cmd[20:70, 5] = 0.1
        q_real = q_cmd.copy()
        q_real[20:70, 5] -= 0.01
        zeros = np.zeros_like(q_cmd)
        run = RunData(
            run_dir=Path("synthetic_equivalent"),
            metadata={"command_rate_hz": 100}, target_joint="RL6_joint",
            repeat_index=1, split_role="fit",
            phase=tuple("synthetic" for _ in range(count)),
            q_cmd_rad=q_cmd, q_real_rad=q_real, dq_real_rad_s=zeros,
            q_trajectory_rad=q_cmd.copy(), dq_trajectory_rad_s=zeros,
            present_pwm_raw=zeros, present_current_a=zeros,
            input_voltage_v=np.full_like(q_cmd, 12.0),
            state_mask=np.ones(count, dtype=bool),
            overrun_mask=np.zeros(count, dtype=bool),
            tx_time_ns=np.arange(count, dtype=np.int64),
            rx_time_ns=np.arange(count, dtype=np.int64),
            q_init_rad=np.zeros(12), dq_init_rad_s=np.zeros(12),
        )
        result = EquivalentReplayResult(q_cmd.copy(), zeros, zeros, zeros, 1)
        parts = equivalent_loss_parts(run, result, config)
        self.assertGreater(parts["plateau"], 0.0)

    def test_joint34_config_ties_left_right_and_fixes_other_joints(self) -> None:
        config = load_joint34_fit_config(
            self.root / "configs" / "joint34_pd.yaml"
        )
        self.assertEqual(
            config.groups,
            {
                "joint3": ("RL3_joint", "LL3_joint"),
                "joint4": ("RL4_joint", "LL4_joint"),
            },
        )
        replay = FixedBaseStatefulJointwisePdReplay(config.replay)
        self.assertEqual(replay.model.nq, 12)
        parameters = joint34_parameters(
            config,
            config.fit_condition,
            {"joint3": 10.0, "joint4": 9.0},
            {"joint3": 0.7, "joint4": 0.8},
        )
        by_name = {
            name: value
            for name, value in zip(MUJOCO_DOF_ORDER, parameters.kp_eff)
        }
        self.assertEqual(by_name["RL3_joint"], by_name["LL3_joint"])
        self.assertEqual(by_name["RL4_joint"], by_name["LL4_joint"])
        self.assertAlmostEqual(by_name["RL1_joint"], 6.0)
        self.assertAlmostEqual(by_name["LL6_joint"], 6.0)

    def test_stateful_play_requires_full_width_after_reversal(self) -> None:
        previous = np.asarray([0.0])
        width = 0.01
        previous = stateful_play_step(previous, np.asarray([0.006]), width)
        np.testing.assert_allclose(previous, [0.001])
        # 반전 직후에는 반대 flank가 아직 닿지 않아 출력이 유지된다.
        previous = stateful_play_step(previous, np.asarray([-0.004]), width)
        np.testing.assert_allclose(previous, [0.001])
        previous = stateful_play_step(previous, np.asarray([-0.010]), width)
        np.testing.assert_allclose(previous, [-0.005])


if __name__ == "__main__":
    unittest.main()
