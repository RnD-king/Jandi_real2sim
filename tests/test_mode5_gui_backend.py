from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import queue
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from jandi_real2sim.mode5.canonical_analysis import Run, primary_fit_sample_counts, real_to_real_repeatability
from jandi_real2sim.mode5.canonical_acquisition import collect_run
from jandi_real2sim.mode5.canonical_attempts import (
    allocate_attempt, inspect_logical_run, select_valid_attempt, write_attempt_selection,
)
from jandi_real2sim.mode5.canonical_config import load_canonical_campaign
from jandi_real2sim.mode5.canonical_trajectories import Sample
from jandi_real2sim.mode5.gui_backend import build_preview, progress_rows, require_physical_confirmation, validated_config_update
from jandi_real2sim.mode5.gui_worker import worker_main


ROOT = Path(__file__).parents[1]


def _metadata(path: Path, valid: bool) -> None:
    (path / "metadata.json").write_text(json.dumps({"valid_flag": valid, "invalid_reason": "" if valid else "test"}))


def _run(repeat: int, offset: float = 0.0) -> Run:
    t = np.arange(20, dtype=float) * 10_000_000
    q = np.linspace(0, 1, 20) + offset
    columns = {
        "host_time_ns": t, "present_position_rad": q,
        "present_velocity_rad_s": np.gradient(q, .01),
        "present_current_A": q * .2,
    }
    return Run(Path(f"r{repeat}"), {"mechanical_configuration": "L1_m250",
        "trajectory": "accelerated_oscillation", "repeat": repeat}, columns)


class AttemptContractTest(unittest.TestCase):
    def test_invalid_retry_allocates_new_immutable_attempt(self):
        with tempfile.TemporaryDirectory() as td:
            logical = Path(td) / "repeat_1"
            first, i1, retry1 = allocate_attempt(logical); _metadata(first, False)
            second, i2, retry2 = allocate_attempt(logical); _metadata(second, True)
            self.assertEqual((i1, retry1), (1, None))
            self.assertEqual((i2, retry2), (2, "attempt_001"))
            self.assertTrue(first.is_dir())
            self.assertEqual(select_valid_attempt(logical), second)

    def test_existing_attempt_metadata_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            logical = Path(td) / "repeat_1"
            first, _, _ = allocate_attempt(logical); _metadata(first, False)
            original = (first / "metadata.json").read_text()
            allocate_attempt(logical)
            self.assertEqual((first / "metadata.json").read_text(), original)

    def test_incomplete_attempt_is_preserved_and_retryable(self):
        with tempfile.TemporaryDirectory() as td:
            logical = Path(td) / "repeat_1"
            first, _, _ = allocate_attempt(logical)
            second, _, _ = allocate_attempt(logical)
            self.assertTrue(first.is_dir()); self.assertTrue(second.is_dir())
            self.assertEqual(len(inspect_logical_run(logical).incomplete_attempts), 2)

    def test_multiple_valid_requires_explicit_selection(self):
        with tempfile.TemporaryDirectory() as td:
            logical = Path(td) / "repeat_1"
            first, _, _ = allocate_attempt(logical); _metadata(first, True)
            second, _, _ = allocate_attempt(logical); _metadata(second, True)
            self.assertEqual(inspect_logical_run(logical).state, "MULTIPLE_VALID_ATTEMPTS")
            with self.assertRaises(ValueError): select_valid_attempt(logical)
            write_attempt_selection(logical, second.name)
            self.assertEqual(select_valid_attempt(logical), second)


class FitValidityAndRepeatabilityTest(unittest.TestCase):
    def test_pwm_saturation_is_the_primary_exclusion(self):
        arrays = (np.arange(4), np.arange(1), np.arange(1), np.arange(4), np.arange(4), np.arange(4),
                  np.array([0, 1, 0, 1], bool), np.array([0, 0, 1, 1], bool))
        counts = primary_fit_sample_counts([(_run(1), arrays)])
        self.assertEqual(counts["total_sample_count"], 4)
        self.assertEqual(counts["normal_sample_count"], 1)
        self.assertEqual(counts["current_saturated_sample_count"], 2)
        self.assertEqual(counts["pwm_saturated_sample_count"], 2)
        self.assertEqual(counts["excluded_from_primary_fit_count"], 2)

    def test_real_to_real_repeatability_has_all_three_pairs(self):
        result = real_to_real_repeatability([_run(1, 0), _run(2, .01), _run(3, -.02)])
        self.assertEqual({row["pair"] for row in result["pairs"]}, {"r1-r2", "r1-r3", "r2-r3"})
        self.assertGreater(result["mean"]["real_repeat_position_rmse_rad"], 0)


class GuiBackendContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_canonical_campaign(ROOT / "configs/mode5/campaign.yaml")

    def test_fixed_l1_l2_and_static_set(self):
        self.assertEqual(self.cfg.geometry["arm_lengths_m"], {"L1": .10, "L2": .15})
        self.assertEqual(len(self.cfg.trajectories["static_calibration"]["static_angles_rad"]), 7)

    def test_preview_needs_no_hardware_configuration_or_port(self):
        cfg = replace(self.cfg, trajectories=copy.deepcopy(self.cfg.trajectories),
                      hardware={**self.cfg.hardware, "encoder_zero_raw": 2048, "direction": 1},
                      safety={**self.cfg.safety, "software_position_min_rad": -2, "software_position_max_rad": 2})
        spec = cfg.trajectories["static_calibration"]
        spec.update(approach_offset_rad=.05, approach_duration_sec=.2,
                    inter_point_transfer_duration_sec=.2, fixed_settling_hold_sec=.1,
                    minimum_settling_sec=.1, averaging_window_sec=.1,
                    maximum_command_speed_rad_s=20,
                    maximum_settled_abs_velocity_rad_s=.1,
                    maximum_settled_position_std_rad=.1,
                    maximum_settled_current_std_A=.1)
        preview = build_preview(cfg, "static")
        self.assertGreater(len(preview.samples), 0)
        self.assertIsNone(cfg.hardware["serial_device"])

    def test_static_and_dynamic_progress_matrix_sizes(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = replace(self.cfg, campaign={**self.cfg.campaign, "id": "x", "output_root": td})
            self.assertEqual(len(progress_rows(cfg, "static")), 36)
            self.assertEqual(len(progress_rows(cfg, "dynamic")), 54)

    def test_backend_safety_gate_runs_before_bus_factory(self):
        called = []
        class NeverBus:
            def __init__(self, _cfg): called.append(True)
        with self.assertRaises(RuntimeError):
            collect_run(self.cfg, "manual", "x", "L1_m250", "manual_hold", 1,
                        [Sample(0, 0.0, "manual_hold", 0.0)],
                        physical_setup_confirmation={"mechanical_configuration": "L1_m250", "confirmed_at": "now"},
                        bus_factory=NeverBus)  # type: ignore[arg-type]
        self.assertEqual(called, [])

    def test_physical_setup_confirmation_is_mandatory(self):
        with self.assertRaises(PermissionError): require_physical_confirmation(None)
        confirmed = {"mechanical_configuration": "L1_m250", "confirmed_at": "now"}
        self.assertEqual(require_physical_confirmation(confirmed), confirmed)

    def test_invalid_config_edit_is_not_persisted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shutil.copytree(ROOT / "configs", root / "configs")
            (root / "README.md").write_text("test")
            (root / "pyproject.toml").write_text("[project]\nname='test'\nversion='0'\n")
            campaign = root / "configs/mode5/campaign.yaml"
            loads = root / "configs/mode5/bench/loads.yaml"
            before = loads.read_text()
            with self.assertRaises(ValueError):
                validated_config_update(campaign, "bench.loads", "loads.m250.measured_mass_kg", -1.0)
            self.assertEqual(loads.read_text(), before)

    def test_fixed_structure_is_not_gui_editable(self):
        with self.assertRaises(ValueError):
            validated_config_update(ROOT / "configs/mode5/campaign.yaml", "bench.geometry", "arm_lengths_m.L1", .2)

    def test_mock_worker_startup_does_not_connect(self):
        for mock in (False, True):
            ctx = mp.get_context("spawn")
            inbox, outbox = ctx.Queue(), ctx.Queue()
            process = ctx.Process(target=worker_main, args=(inbox, outbox), kwargs={"mock": mock})
            process.start()
            first = outbox.get(timeout=5)
            self.assertEqual(first["type"], "worker_ready")
            inbox.put({"action": "shutdown"}); process.join(timeout=5)
            self.assertFalse(process.is_alive())

    def test_mock_torque_off_message_contract(self):
        ctx = mp.get_context("spawn")
        inbox, outbox = ctx.Queue(), ctx.Queue()
        process = ctx.Process(target=worker_main, args=(inbox, outbox), kwargs={"mock": True})
        process.start(); outbox.get(timeout=5)
        inbox.put({"action": "torque_off", "config": str(ROOT / "configs/mode5/campaign.yaml")})
        self.assertEqual(outbox.get(timeout=5)["type"], "torque_off_complete")
        inbox.put({"action": "shutdown"}); process.join(timeout=5)

    def test_active_mock_run_torque_off_marks_operator_abort(self):
        ctx = mp.get_context("spawn")
        inbox, outbox = ctx.Queue(), ctx.Queue()
        process = ctx.Process(target=worker_main, args=(inbox, outbox), kwargs={"mock": True})
        process.start(); outbox.get(timeout=5)
        inbox.put({"action": "run", "duration_sec": 2.0})
        while outbox.get(timeout=5)["type"] != "telemetry": pass
        inbox.put({"action": "torque_off"})
        message = outbox.get(timeout=5)
        while message["type"] != "aborted": message = outbox.get(timeout=5)
        self.assertEqual(message["reason"], "operator_abort")
        inbox.put({"action": "shutdown"}); process.join(timeout=5)

    def test_mock_manual_style_run_is_not_saved_to_canonical_data(self):
        ctx = mp.get_context("spawn")
        inbox, outbox = ctx.Queue(), ctx.Queue()
        process = ctx.Process(target=worker_main, args=(inbox, outbox), kwargs={"mock": True})
        process.start(); outbox.get(timeout=5)
        inbox.put({"action": "run", "duration_sec": .02})
        messages = []
        while not any(item.get("type") == "completed" for item in messages):
            messages.append(outbox.get(timeout=5))
        completed = next(item for item in messages if item["type"] == "completed")
        self.assertFalse(completed["saved"])
        inbox.put({"action": "shutdown"}); process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
