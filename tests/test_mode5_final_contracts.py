from __future__ import annotations

import copy
import csv
import inspect
import json
import math
import tempfile
import unittest
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from jandi_real2sim.mode5 import canonical_acquisition, canonical_bus, gui_app
from jandi_real2sim.mode5.canonical_acquisition import (
    TELEMETRY_FIELDS, _run_samples, _verify_physical_setup,
    _verify_campaign_freeze, _write_campaign_freeze, post_run_integrity_check,
)
from jandi_real2sim.mode5.canonical_analysis import Run, assert_dataset_compatible
from jandi_real2sim.mode5.canonical_bus import CanonicalMode5Bus, State
from jandi_real2sim.mode5.canonical_config import load_canonical_campaign
from jandi_real2sim.mode5.canonical_model import build_model
from jandi_real2sim.mode5.canonical_trajectories import Sample, static_run_specs
from jandi_real2sim.mode5.gui_backend import EDITABLE_FIELDS

from tests.test_mode5_canonical_contracts import resolved_cfg


ROOT = Path(__file__).parents[1]


class Clock:
    def __init__(self):
        self.now = 0

    def monotonic_ns(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


class Writer:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(dict(row))


class LoopBus:
    def __init__(self, cfg, clock, *, stale=False, read_cost_ns=0):
        self.cfg, self.clock = cfg, clock
        self.goal_raw = cfg.rad_to_raw(0.0)
        self.writes = 0
        self.reads = 0
        self.stale = stale
        self.read_cost_ns = read_cost_ns

    def write_goal_rad_no_response(self, goal):
        before = self.clock.now
        self.goal_raw = self.cfg.rad_to_raw(goal)
        self.writes += 1
        return canonical_bus.GoalWrite(self.goal_raw, before, self.clock.now)

    def read_state(self):
        self.clock.now += self.read_cost_ns
        self.reads += 1
        goal = self.goal_raw - 1 if self.stale and self.reads >= 2 else self.goal_raw
        return State(goal, self.reads, 0, 0, 0, 0, 0, self.goal_raw,
                     0, self.goal_raw, 120, 30)

    def read_hardware_error(self):
        return 0


def _raw_row(index: int, time_ns: int, tick: int) -> dict[str, object]:
    row: dict[str, object] = {name: 0 for name in TELEMETRY_FIELDS}
    row.update({
        "sample_index": index, "target_seq": index // 2,
        "target_update_event": int(index % 2 == 0), "bus_write_seq": index,
        "bus_write_event": 1, "command_seq": index // 2,
        "command_event": int(index % 2 == 0), "command_tx_before_ns": time_ns - 3,
        "command_tx_after_ns": time_ns - 2, "state_read_before_ns": time_ns - 1,
        "state_read_after_ns": time_ns + 1, "state_time_ns": time_ns,
        "host_time_ns": time_ns + 1, "host_time_sec": time_ns * 1e-9,
        "goal_position_rad": 0.0, "present_position_rad": 0.0,
        "present_velocity_rad_s": 0.0, "present_current_A": 0.0,
        "present_pwm_fraction": 0.0, "input_voltage_V": 12.0,
        "realtime_tick_unwrapped_ms": tick, "fit_eligible": 1,
    })
    return row


class TimingAndRawContractTest(unittest.TestCase):
    def test_main_loop_is_50_hz_targets_and_100_hz_write_state(self):
        cfg, clock = resolved_cfg(), Clock()
        bus, writer = LoopBus(cfg, clock), Writer()
        samples = [Sample(i, i / 50, "hold", 0.01 * i) for i in range(5)]
        with patch.object(canonical_acquisition.time, "monotonic_ns", clock.monotonic_ns), \
             patch.object(canonical_acquisition.time, "sleep", clock.sleep):
            stats = _run_samples(bus, cfg, samples, writer, Writer())
        self.assertEqual((len(stats.target_update_ns), len(stats.bus_write_ns), len(stats.state_time_ns)), (5, 10, 10))
        self.assertEqual(sum(int(row["target_update_event"]) for row in writer.rows), 5)
        self.assertEqual(sum(int(row["bus_write_event"]) for row in writer.rows), 10)

    def test_state_midpoint_is_inside_read_bracket(self):
        cfg, clock = resolved_cfg(), Clock()
        bus, writer = LoopBus(cfg, clock, read_cost_ns=2000), Writer()
        with patch.object(canonical_acquisition.time, "monotonic_ns", clock.monotonic_ns), \
             patch.object(canonical_acquisition.time, "sleep", clock.sleep):
            _run_samples(bus, cfg, [Sample(0, 0, "hold", 0)], writer, Writer())
        for row in writer.rows:
            self.assertLessEqual(int(row["state_read_before_ns"]), int(row["state_time_ns"]))
            self.assertLessEqual(int(row["state_time_ns"]), int(row["state_read_after_ns"]))

    def test_goal_readback_grace_then_abort_preserves_row(self):
        cfg, clock = resolved_cfg(), Clock()
        bus, writer = LoopBus(cfg, clock, stale=True), Writer()
        with patch.object(canonical_acquisition.time, "monotonic_ns", clock.monotonic_ns), \
             patch.object(canonical_acquisition.time, "sleep", clock.sleep), \
             self.assertRaisesRegex(RuntimeError, "GOAL_READBACK_MISMATCH"):
            _run_samples(bus, cfg, [Sample(0, 0, "hold", .1)], writer, Writer())
        self.assertEqual(len(writer.rows), 2)
        self.assertEqual(int(writer.rows[-1]["goal_readback_mismatch"]), 1)

    def test_severe_overrun_row_is_written_and_no_catchup_occurs(self):
        cfg, clock = resolved_cfg(), Clock()
        timing = {**cfg.timing, "severe_overrun_threshold_sec": .01}
        cfg = replace(cfg, timing=timing)
        bus, writer = LoopBus(cfg, clock, read_cost_ns=25_000_000), Writer()
        with patch.object(canonical_acquisition.time, "monotonic_ns", clock.monotonic_ns), \
             patch.object(canonical_acquisition.time, "sleep", clock.sleep), \
             self.assertRaisesRegex(RuntimeError, "TIMING_INVALID"):
            _run_samples(bus, cfg, [Sample(i, i / 50, "hold", 0) for i in range(5)], writer, Writer())
        self.assertEqual(bus.writes, 1)
        self.assertEqual(len(writer.rows), 1)
        self.assertEqual(int(writer.rows[0]["timing_invalid"]), 1)

    def test_tx_timestamps_bracket_actual_txpacket_call(self):
        cfg = resolved_cfg()
        bus = CanonicalMode5Bus(cfg)
        calls = []

        class GoalWriter:
            def clearParam(self): pass
            def addParam(self, *_): return True
            def txPacket(self): calls.append("tx"); return 0

        bus._port = object(); bus._packet = SimpleNamespace(getTxRxResult=lambda _: "")
        bus._sdk = SimpleNamespace(COMM_SUCCESS=0); bus._goal_writer = GoalWriter()
        with patch.object(canonical_bus.time, "monotonic_ns", side_effect=[10, 20]):
            result = bus.write_goal_rad_no_response(0.0)
        self.assertEqual(calls, ["tx"])
        self.assertEqual((result.tx_before_ns, result.tx_after_ns), (10, 20))

    def test_configuration_snapshot_contains_required_diagnostics(self):
        bus = CanonicalMode5Bus(resolved_cfg())
        bus.read = lambda name: 0  # type: ignore[method-assign]
        snapshot = bus.read_configuration_snapshot()
        for name in ("firmware_version", "return_delay_time_raw", "homing_offset_raw",
                     "status_return_level_raw", "hardware_error"):
            self.assertIn(name, snapshot)


class IntegrityFrozenAndCompatibilityTest(unittest.TestCase):
    def _integrity(self, rows):
        cfg = resolved_cfg()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            with (path / "telemetry.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=TELEMETRY_FIELDS)
                writer.writeheader(); writer.writerows(rows)
            count = len(rows)
            stats = canonical_acquisition.AcquisitionStats(
                count, tuple(int(r["command_tx_after_ns"]) for r in rows if int(r["target_update_event"])),
                tuple(int(r["state_time_ns"]) for r in rows), tuple(0 for _ in rows),
                tuple(int(r["command_tx_after_ns"]) for r in rows), tuple(2 for _ in rows),
            )
            return post_run_integrity_check(path, cfg, stats)

    def test_post_run_integrity_accepts_well_formed_raw(self):
        self.assertTrue(self._integrity([_raw_row(i, 10_000_000 * (i + 1), 10 * i) for i in range(4)])["passed"])

    def test_post_run_integrity_rejects_nan(self):
        rows = [_raw_row(i, 10_000_000 * (i + 1), 10 * i) for i in range(4)]
        rows[2]["present_current_A"] = math.nan
        result = self._integrity(rows)
        self.assertFalse(result["passed"]); self.assertIn("non-finite telemetry", result["errors"])

    def test_post_run_integrity_rejects_gross_tick_anomaly(self):
        rows = [_raw_row(i, 10_000_000 * (i + 1), 1000 * i) for i in range(4)]
        result = self._integrity(rows)
        self.assertFalse(result["passed"])
        self.assertTrue(any("Realtime Tick" in error for error in result["errors"]))

    def test_post_run_integrity_rejects_truncated_planned_run(self):
        cfg = resolved_cfg()
        rows = [_raw_row(i, 10_000_000 * (i + 1), 10 * i) for i in range(2)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            with (path / "telemetry.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=TELEMETRY_FIELDS)
                writer.writeheader(); writer.writerows(rows)
            stats = canonical_acquisition.AcquisitionStats(
                2, (rows[0]["command_tx_after_ns"],),
                tuple(int(r["state_time_ns"]) for r in rows), (0, 0),
                tuple(int(r["command_tx_after_ns"]) for r in rows), (2, 2),
            )
            result = post_run_integrity_check(path, cfg, stats, expected_sample_count=4)
        self.assertFalse(result["passed"])
        self.assertTrue(any("planned_sample_count" in error for error in result["errors"]))

    def test_physical_confirmation_mismatch_blocks(self):
        cfg = resolved_cfg()
        with self.assertRaises(PermissionError):
            _verify_physical_setup(cfg, "L1_m250", {"mechanical_configuration": "L2_m250", "arm_length_m": .15, "measured_mass_kg": .25})

    def test_pilot_and_delay_mechanical_selection_cannot_bypass_canonical_config(self):
        cfg = resolved_cfg()
        with self.assertRaisesRegex(PermissionError, "pilot mechanical mismatch"):
            canonical_acquisition.collect_run(
                cfg, "pilot", "pilot/run_1", "L2_m250", "pilot", 1,
                [Sample(0, 0, "pilot", 0)], bus_factory=object,
            )
        with self.assertRaisesRegex(PermissionError, "delay mechanical mismatch"):
            canonical_acquisition.collect_run(
                cfg, "delay", "delay/probe_1", "L2_m250", "delay_probe", 1,
                [Sample(0, 0, "delay", 0)], bus_factory=object,
            )

    def test_campaign_freeze_detects_later_manifest_change(self):
        cfg = resolved_cfg()
        with tempfile.TemporaryDirectory() as td:
            cfg = replace(cfg, campaign={**cfg.campaign, "output_root": td})
            _write_campaign_freeze(cfg, Path(td) / "first")
            freeze_path = Path(td) / "test" / "campaign_freeze.json"
            frozen = json.loads(freeze_path.read_text())
            frozen["critical_config_manifest_sha256"] = "simulated-later-config-change"
            freeze_path.write_text(json.dumps(frozen))
            with self.assertRaisesRegex(RuntimeError, "freeze"):
                _verify_campaign_freeze(cfg)

    def test_incompatible_frozen_controller_blocks_fit(self):
        cfg = resolved_cfg()
        resolved = {"hardware": cfg.hardware, "controller": cfg.registers, "timing": cfg.timing}
        meta = {"mechanical_configuration": "L1_m250", "trajectory": "x", "repeat": 1, "resolved": resolved}
        first = Run(Path("a"), meta, {})
        changed = copy.deepcopy(resolved); changed["controller"]["position_p_gain"] += 1
        second = Run(Path("b"), {**meta, "repeat": 2, "resolved": changed}, {})
        with self.assertRaisesRegex(ValueError, "incompatible"):
            assert_dataset_compatible([first, second])

    def test_frozen_geometry_drives_old_run_model(self):
        cfg = resolved_cfg()
        frozen = {"geometry": copy.deepcopy(cfg.geometry), "loads": copy.deepcopy(cfg.loads),
                  "controller": copy.deepcopy(cfg.registers)}
        params = {"armature_kg_m2": .01, "coulomb_friction_Nm": .01,
                  "viscous_friction_Nm_s_per_rad": .01}
        changed = replace(cfg, geometry={**cfg.geometry, "arm_mass_kg": 9.0})
        old = build_model(cfg, "L1_m250", params, .001, frozen)
        replayed = build_model(changed, "L1_m250", params, .001, frozen)
        np.testing.assert_allclose(old.body_mass, replayed.body_mass)


class MatrixGuiAndFitPolicyTest(unittest.TestCase):
    def test_static_matrix_has_36_sweeps_and_252_plateaus(self):
        cfg = load_canonical_campaign(ROOT / "configs/mode5/campaign.yaml")
        self.assertEqual(len(static_run_specs(cfg)), 36)
        self.assertEqual(len(static_run_specs(cfg)) * len(cfg.trajectories["static_calibration"]["static_angles_rad"]), 252)

    def test_gui_exposes_all_workflow_modes_and_attempt_controls(self):
        source = inspect.getsource(gui_app.MainWindow)
        for text in ("Manual / Play", "Pilot", "Static Calibration", "Delay Calibration",
                     "Main Dynamic Dataset", "FIT", "VALIDATE", "GENERATE REPORT",
                     "Retry selected logical run", "Select accepted attempt", "Keep temporary log"):
            self.assertIn(text, source)

    def test_offscreen_gui_startup_does_not_connect_or_open_serial(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        class FakeProcess:
            def __init__(self): self.started = False
            def start(self): self.started = True
            def join(self, timeout=None): pass
            def is_alive(self): return False
            def terminate(self): pass

        fake = FakeProcess()
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with patch.object(gui_app.mp, "Process", return_value=fake):
            window = gui_app.MainWindow(ROOT / "configs/mode5/campaign.yaml", mock=True)
            self.assertTrue(fake.started)
            self.assertFalse(window.connected)
            self.assertIn("Not connected", window.register_status.text())
            window.close()
        app.processEvents()

    def test_fixed_canonical_constants_are_not_editable(self):
        editable = {field for fields in EDITABLE_FIELDS.values() for field in fields}
        for field in ("arm_lengths_m.L1", "arm_lengths_m.L2", "parameters.static_angles_rad",
                      "timing.target_generation_rate_hz", "timing.bus_write_rate_hz",
                      "timing.state_read_rate_hz"):
            self.assertNotIn(field, editable)

    def test_stage_d_policy_retains_current_and_excludes_pwm(self):
        source = inspect.getsource(canonical_acquisition) + inspect.getsource(__import__(
            "jandi_real2sim.mode5.canonical_analysis", fromlist=["fit"]))
        self.assertIn("primary = ~pwm_sat & eligible", source)
        self.assertNotIn("primary = ~current_sat", source)

    def test_static_and_ad_exclude_both_saturations(self):
        from jandi_real2sim.mode5 import canonical_analysis
        static_source = inspect.getsource(canonical_analysis.estimate_static)
        ad_source = inspect.getsource(canonical_analysis.estimate_ad)
        for source in (static_source, ad_source):
            self.assertIn("current_saturated", source)
            self.assertIn("pwm_saturated", source)


if __name__ == "__main__":
    unittest.main()
