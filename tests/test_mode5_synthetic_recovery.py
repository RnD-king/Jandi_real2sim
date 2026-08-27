from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from jandi_real2sim.mode5.canonical_analysis import Run, estimate_delay, estimate_static
from jandi_real2sim.mode5.canonical_model import replay
from tests.test_mode5_canonical_contracts import resolved_cfg


def static_runs(noise: float) -> list[Run]:
    cfg = resolved_cfg()
    rng = np.random.default_rng(42)
    ktau, ap = 1.4, 4.0
    result = []
    names = ("L1_m250", "L1_m500", "L1_m750", "L2_m250", "L2_m500", "L2_m750")
    for run_index, mechanical in enumerate(names):
        approach = "approach_positive" if run_index % 2 == 0 else "approach_negative"
        phases, q_values, goals, currents = [], [], [], []
        for point, q_nominal in enumerate((-.45, -.2, .15, .4)):
            moment = 9.80665 * (.1 * .08 + cfg.load_mass_kg(mechanical) * cfg.arm_length_m(mechanical))
            torque = moment * np.sin(q_nominal)
            current = torque / ktau
            goal = q_nominal + current / ap
            for _ in range(20):
                phases.append(f"point_{point}_averaging")
                q_values.append(q_nominal + rng.normal(0, noise))
                goals.append(goal)
                currents.append(current + rng.normal(0, noise))
        n = len(phases)
        columns = {
            "phase": np.asarray(phases, dtype=object),
            "present_position_rad": np.asarray(q_values),
            "goal_position_rad": np.asarray(goals),
            "present_current_A": np.asarray(currents),
            "present_velocity_rad_s": np.zeros(n),
            "current_saturated": np.zeros(n), "pwm_saturated": np.zeros(n),
        }
        result.append(Run(Path(f"run_{run_index}_{approach}"),
                          {"mechanical_configuration": mechanical,
                           "trajectory": "static_calibration", "repeat": 1}, columns))
    return result


def delay_run(delay_s: float) -> Run:
    t = np.arange(0, 2.0, .001)
    event_times = np.asarray([0., .3, .7, 1.1, 1.5])
    goals = np.asarray([0., .1, 0., -.1, 0.])
    event = np.zeros(len(t))
    full_goal = np.zeros(len(t))
    tx = np.zeros(len(t))
    for index, event_time in enumerate(event_times):
        sample = int(round(event_time / .001))
        event[sample] = 1
        full_goal[sample:] = goals[index]
        tx[sample:] = int(round(event_time * 1e9))
    delayed = np.zeros(len(t))
    for index, now in enumerate(t):
        eligible = np.flatnonzero(event_times <= now - delay_s + 1e-12)
        delayed[index] = goals[eligible[-1]] if len(eligible) else goals[0]
    return Run(Path("delay"), {"mechanical_configuration": "L1_m250",
        "trajectory": "delay_probe", "repeat": 1}, {
        "host_time_ns": t * 1e9, "command_tx_after_ns": tx,
        "command_event": event, "goal_position_rad": full_goal,
        "present_current_A": delayed,
    })


class SyntheticRecoveryTest(unittest.TestCase):
    def test_static_recovery_noise_free_and_low_noise(self):
        cfg = resolved_cfg()
        bootstrap = {"repeat_count": 40, "random_seed": 3,
                     "condition_number_warning_threshold": 1e8}
        for noise, tolerance in ((0.0, 1e-8), (2e-4, .03)):
            result = estimate_static(cfg, static_runs(noise), bootstrap)
            self.assertAlmostEqual(result["Ktau_eff_prior_Nm_per_A"], 1.4, delta=tolerance)
            self.assertAlmostEqual(result["aP_prior_A_per_rad"], 4.0, delta=tolerance)
            self.assertEqual(result["accepted_plateau_count"], 24)
            self.assertEqual(result["rejected_plateau_count"], 0)

    def test_delay_recovery(self):
        cfg = resolved_cfg()
        cfg.trajectories["delay_probe"].update(onset_current_threshold_A=.02,
            pre_event_baseline_sec=.1, response_search_sec=.1)
        result = estimate_delay(cfg, delay_run(.012))
        self.assertAlmostEqual(result["delay_median_s"], .012, delta=.0011)
        self.assertAlmostEqual(result["sampling_resolution_s"], .001, delta=1e-9)

    def test_dynamic_parameter_recovery_noise_free_and_low_noise(self):
        cfg = resolved_cfg()
        t = np.arange(0, 2.5, .02)
        command_t = np.arange(0, 2.5, .01)
        command = .12*np.sin(2*np.pi*.7*command_t) + .035*np.sin(2*np.pi*2.1*command_t)
        fixed = {"aP_A_per_rad": 5.0, "Ktau_eff_Nm_per_A": 1.4, "delay_s": .012}
        names = ("aD_A_s_per_rad", "armature_kg_m2", "coulomb_friction_Nm",
                 "viscous_friction_Nm_s_per_rad")
        truth = np.asarray([.35, .006, .025, .045])
        true_params = {**fixed, **dict(zip(names, truth))}
        exact = replay(cfg, "L1_m500", t, command_t, command, 0., 0., true_params, .002)
        for noise, tolerance in ((0.0, .03), (2e-5, .20)):
            rng = np.random.default_rng(9)
            q = exact.q_rad + rng.normal(0, noise, len(t))
            qd = exact.qd_rad_s + rng.normal(0, noise*5, len(t))
            current = exact.current_A + rng.normal(0, noise*2, len(t))

            def residual(x):
                sim = replay(cfg, "L1_m500", t, command_t, command, q[0], qd[0],
                             {**fixed, **dict(zip(names, x))}, .002)
                return np.concatenate(((sim.q_rad-q)/.02, (sim.qd_rad_s-qd)/.2,
                                       (sim.current_A-current)/.1))

            fit = least_squares(residual, [.25, .01, .015, .08],
                                bounds=([0, .001, 0, 0], [1, .03, .1, .2]), max_nfev=80)
            relative = np.abs((fit.x-truth)/truth)
            self.assertTrue(np.all(relative < tolerance), (noise, fit.x, relative))


if __name__ == "__main__":
    unittest.main()
