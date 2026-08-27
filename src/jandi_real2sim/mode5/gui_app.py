"""PySide6/pyqtgraph desktop interface for the canonical Mode-5 backend."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import queue
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .canonical_config import load_canonical_campaign
from .canonical_acquisition import cooldown_remaining_sec
from .canonical_trajectories import dynamic_run_specs, static_run_specs
from .gui_backend import EDITABLE_FIELDS, build_preview, progress_rows, validated_config_update
from .gui_worker import worker_main
from .spec import CONFIRMATIONS, DEFAULT_CAMPAIGN, MAIN_TRAJECTORIES, MECHANICAL_CONFIGURATIONS


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: Path, mock: bool = False):
        super().__init__()
        self.config_path = config.resolve()
        self.mock = mock
        self.cfg = load_canonical_campaign(self.config_path)
        self.inbox: mp.Queue = mp.Queue()
        self.outbox: mp.Queue = mp.Queue()
        self.worker = mp.Process(target=worker_main, args=(self.inbox, self.outbox),
                                 kwargs={"mock": mock}, daemon=True)
        self.worker.start()  # worker startup has no serial/open action
        self.connected = False
        self.active_run = False
        self.buffers = {name: [] for name in ("t", "goal", "q", "qd", "current", "pwm")}
        self.setWindowTitle("MX-106 Mode 5 Real2Sim Experiment" + (" [EXPLICIT MOCK]" if mock else ""))
        self.resize(1500, 920)
        self._build()
        self._refresh_from_config()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._drain_worker)
        self.timer.start(33)

    def _build(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        split = QtWidgets.QSplitter()
        layout.addWidget(split, 1)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        split.addWidget(left)
        split.setStretchFactor(1, 1)

        connection = QtWidgets.QGroupBox("Connection (no auto-connect)")
        form = QtWidgets.QFormLayout(connection)
        self.port = QtWidgets.QLabel()
        self.baud = QtWidgets.QLabel()
        self.motor = QtWidgets.QLabel()
        self.register_status = QtWidgets.QLabel("Not connected — no hardware readback")
        self.register_status.setWordWrap(True)
        self.register_table = QtWidgets.QTableWidget(0, 2)
        self.register_table.setHorizontalHeaderLabels(("Register", "Readback"))
        self.register_table.horizontalHeader().setStretchLastSection(True)
        self.register_table.setMinimumHeight(180)
        form.addRow("Serial", self.port); form.addRow("Baudrate", self.baud); form.addRow("Motor ID", self.motor)
        form.addRow("Motor readback", self.register_status)
        form.addRow(self.register_table)
        buttons = QtWidgets.QHBoxLayout()
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.disconnect_button = QtWidgets.QPushButton("Disconnect")
        buttons.addWidget(self.connect_button); buttons.addWidget(self.disconnect_button)
        form.addRow(buttons)
        self.connect_button.clicked.connect(lambda: self.inbox.put({"action": "connect", "config": str(self.config_path)}))
        self.disconnect_button.clicked.connect(lambda: self.inbox.put({"action": "disconnect"}))
        left_layout.addWidget(connection)

        setup = QtWidgets.QGroupBox("Bench Setup")
        setup_form = QtWidgets.QFormLayout(setup)
        self.mechanical = QtWidgets.QComboBox(); self.mechanical.addItems(MECHANICAL_CONFIGURATIONS)
        self.measured_mass = QtWidgets.QLineEdit()
        self.measured_mass.setPlaceholderText("REQUIRED measured kg (blank keeps null)")
        self.save_mass = QtWidgets.QPushButton("Validate & Save measured mass")
        self.config_editor = QtWidgets.QPushButton("Validated config editor…")
        self.geometry_status = QtWidgets.QLabel(); self.geometry_status.setWordWrap(True)
        setup_form.addRow("Arm/load", self.mechanical); setup_form.addRow("Fixed geometry", self.geometry_status); setup_form.addRow("Measured load", self.measured_mass); setup_form.addRow(self.save_mass); setup_form.addRow(self.config_editor)
        self.mechanical.currentTextChanged.connect(self._show_mass)
        self.save_mass.clicked.connect(self._save_measured_mass)
        self.config_editor.clicked.connect(self._edit_config)
        left_layout.addWidget(setup)

        experiment = QtWidgets.QGroupBox("Experiment Mode")
        exp_form = QtWidgets.QFormLayout(experiment)
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(("Manual / Play", "Pilot", "Static Calibration", "Delay Calibration", "Main Dynamic Dataset"))
        self.trajectory = QtWidgets.QComboBox(); self.trajectory.addItems(MAIN_TRAJECTORIES)
        self.approach = QtWidgets.QComboBox(); self.approach.addItems(("approach_positive", "approach_negative"))
        self.repeat = QtWidgets.QComboBox(); self.repeat.addItems(("1", "2", "3"))
        self.override = QtWidgets.QCheckBox("Override planned order")
        self.override_reason = QtWidgets.QLineEdit(); self.override_reason.setPlaceholderText("required reason")
        self.manual_center = QtWidgets.QDoubleSpinBox(); self.manual_center.setRange(-6.2, 6.2)
        self.manual_target = QtWidgets.QComboBox(); self.manual_target.addItems(("Hold", "Step", "Sine"))
        self.manual_amplitude = QtWidgets.QDoubleSpinBox(); self.manual_amplitude.setRange(0, 3); self.manual_amplitude.setValue(.1)
        self.manual_frequency = QtWidgets.QDoubleSpinBox(); self.manual_frequency.setRange(.01, 20); self.manual_frequency.setValue(.5)
        self.manual_duration = QtWidgets.QDoubleSpinBox(); self.manual_duration.setRange(.1, 120); self.manual_duration.setValue(3)
        self.keep_manual_log = QtWidgets.QCheckBox("Keep temporary log in data/temp/manual")
        self.keep_manual_log.setChecked(True)
        for label, widget in (("Mode", self.mode), ("Trajectory", self.trajectory), ("Approach", self.approach), ("Repeat", self.repeat),
                              ("Manual target", self.manual_target), ("Manual center [rad]", self.manual_center), ("Manual amplitude [rad]", self.manual_amplitude),
                              ("Manual frequency [Hz]", self.manual_frequency), ("Manual duration [s]", self.manual_duration),
                              ("Manual logging", self.keep_manual_log),
                              ("Order", self.override), ("Override reason", self.override_reason)):
            exp_form.addRow(label, widget)
        left_layout.addWidget(experiment)

        campaign = QtWidgets.QGroupBox("Dataset / Campaign")
        campaign_form = QtWidgets.QFormLayout(campaign)
        self.campaign_id = QtWidgets.QLineEdit()
        self.save_campaign = QtWidgets.QPushButton("Validate & set campaign")
        self.next_run = QtWidgets.QLabel()
        campaign_form.addRow("Campaign", self.campaign_id); campaign_form.addRow(self.save_campaign); campaign_form.addRow("NEXT RUN", self.next_run)
        self.save_campaign.clicked.connect(self._save_campaign)
        left_layout.addWidget(campaign)

        approvals = QtWidgets.QGroupBox("Human approvals")
        approvals_form = QtWidgets.QFormLayout(approvals)
        self.approve_pilot = QtWidgets.QPushButton("PASS latest valid pilot")
        self.fail_pilot = QtWidgets.QPushButton("FAIL pilot")
        self.ack_warmup = QtWidgets.QPushButton("ACKNOWLEDGE WARMUP")
        self.approval_status = QtWidgets.QLabel(); self.approval_status.setWordWrap(True)
        approvals_form.addRow(self.approve_pilot, self.fail_pilot); approvals_form.addRow(self.ack_warmup)
        approvals_form.addRow("Status / procedure", self.approval_status)
        self.approve_pilot.clicked.connect(self._approve_pilot)
        self.fail_pilot.clicked.connect(self._fail_pilot)
        self.ack_warmup.clicked.connect(self._acknowledge_warmup)
        left_layout.addWidget(approvals)
        left_layout.addStretch(1)

        right = QtWidgets.QTabWidget(); split.addWidget(right)
        telemetry = QtWidgets.QWidget(); right.addTab(telemetry, "Live telemetry")
        grid = QtWidgets.QGridLayout(telemetry)
        self.plots = {}
        for index, (key, title) in enumerate((("position", "Goal + Present Position"), ("current", "Present Current"), ("velocity", "Present Velocity"), ("pwm", "Present PWM"))):
            plot = pg.PlotWidget(title=title); plot.showGrid(x=True, y=True, alpha=.25)
            grid.addWidget(plot, index // 2, index % 2); self.plots[key] = plot
        self.status = QtWidgets.QPlainTextEdit(); self.status.setReadOnly(True); grid.addWidget(self.status, 2, 0, 1, 2)
        self.live_numbers = QtWidgets.QLabel("Voltage — | Temperature — | Current/PWM saturation — | Telemetry rate —")
        self.live_state = QtWidgets.QLabel(
            "Goal — | Position — | Velocity — | Current — | PWM — | Tick — | Moving — | Status —"
        )
        grid.addWidget(self.live_numbers, 3, 0, 1, 2)
        grid.addWidget(self.live_state, 4, 0, 1, 2)

        progress = QtWidgets.QWidget(); right.addTab(progress, "Progress")
        progress_layout = QtWidgets.QVBoxLayout(progress)
        self.progress_summary = QtWidgets.QLabel()
        self.progress_table = QtWidgets.QTableWidget()
        attempt_buttons = QtWidgets.QHBoxLayout()
        self.retry_selected = QtWidgets.QPushButton("Retry selected logical run")
        self.select_attempt = QtWidgets.QPushButton("Select accepted attempt…")
        attempt_buttons.addWidget(self.retry_selected); attempt_buttons.addWidget(self.select_attempt)
        progress_layout.addWidget(self.progress_summary); progress_layout.addWidget(self.progress_table)
        progress_layout.addLayout(attempt_buttons)
        self.retry_selected.clicked.connect(self._prepare_retry)
        self.select_attempt.clicked.connect(self._select_accepted_attempt)

        analysis = QtWidgets.QWidget(); right.addTab(analysis, "Fit / Validate / Report")
        analysis_layout = QtWidgets.QFormLayout(analysis)
        self.result_dir = QtWidgets.QLineEdit(); self.result_dir.setPlaceholderText("results/<timestamp>_mode5_m1")
        self.fit_button = QtWidgets.QPushButton("FIT (Static → Delay → aD → Stage D)")
        self.validate_button = QtWidgets.QPushButton("VALIDATE 9-run holdout")
        self.report_button = QtWidgets.QPushButton("GENERATE REPORT")
        analysis_layout.addRow("Result directory", self.result_dir)
        analysis_layout.addRow(self.fit_button); analysis_layout.addRow(self.validate_button); analysis_layout.addRow(self.report_button)
        self.fit_button.clicked.connect(lambda: self.inbox.put({"action": "fit", "config": str(self.config_path)}))
        self.validate_button.clicked.connect(lambda: self._analysis_action("validate"))
        self.report_button.clicked.connect(lambda: self._analysis_action("report"))

        bottom = QtWidgets.QHBoxLayout(); layout.addLayout(bottom)
        self.preview_button = QtWidgets.QPushButton("PREVIEW (NO HARDWARE)")
        self.run_button = QtWidgets.QPushButton("RUN + SAVE")
        self.torque_off = QtWidgets.QPushButton("TORQUE OFF")
        self.torque_off.setStyleSheet("font-weight:bold; color:white; background:#b00020; min-height:44px")
        bottom.addWidget(self.preview_button); bottom.addWidget(self.run_button); bottom.addStretch(1); bottom.addWidget(self.torque_off)
        self.preview_button.clicked.connect(self.preview)
        self.run_button.clicked.connect(self.run)
        self.torque_off.clicked.connect(self.request_torque_off)
        self.mode.currentIndexChanged.connect(self.refresh_progress)
        self.mechanical.currentTextChanged.connect(self.refresh_progress)
        self.trajectory.currentTextChanged.connect(self.refresh_progress)
        self.repeat.currentTextChanged.connect(self.refresh_progress)

    def _refresh_from_config(self) -> None:
        self.cfg = load_canonical_campaign(self.config_path)
        self.port.setText(str(self.cfg.hardware.get("serial_device")))
        self.baud.setText(str(self.cfg.hardware.get("baudrate")))
        self.motor.setText(str(self.cfg.hardware.get("motor_id")))
        self.campaign_id.setText(str(self.cfg.campaign_id or ""))
        self.geometry_status.setText(
            f"L1={self.cfg.geometry['arm_lengths_m']['L1']*1000:.0f} mm, L2={self.cfg.geometry['arm_lengths_m']['L2']*1000:.0f} mm\n"
            f"arm mass={self.cfg.geometry.get('arm_mass_kg')}, COM={self.cfg.geometry.get('arm_com_radius_m')}, inertia={self.cfg.geometry.get('arm_inertia_kg_m2')} ({self.cfg.geometry.get('arm_inertia_reference')})\n"
            "q=0: physical upright | gravity_zero=0\nStatic angles fixed: -90, -60, -30, 0, +30, +60, +90 deg\n"
            "Target 50 Hz | Bus write 100 Hz | State 100 Hz"
        )
        self.approval_status.setText(
            f"Pilot passed={self.cfg.approval.get('pilot_passed')} | "
            f"reference={self.cfg.approval.get('pilot_run_reference')} | "
            f"warmup ack={self.cfg.approval.get('warmup_acknowledged_at')}\n"
            f"Warmup procedure: {self.cfg.safety.get('warmup_procedure')}"
        )
        self._show_mass(); self.refresh_progress()

    def _show_mass(self) -> None:
        if not hasattr(self, "measured_mass"): return
        load = self.cfg.configuration(self.mechanical.currentText()).load
        value = self.cfg.loads[load].get("measured_mass_kg")
        self.measured_mass.setText("" if value is None else f"{float(value):.9g}")

    def _save_measured_mass(self) -> None:
        load = self.cfg.configuration(self.mechanical.currentText()).load
        try:
            text = self.measured_mass.text().strip()
            value = None if not text else float(text)
            validated_config_update(self.config_path, "bench.loads", f"loads.{load}.measured_mass_kg", value)
            self._refresh_from_config(); self._log("Validated measured mass saved to canonical YAML.")
        except BaseException as exc:
            QtWidgets.QMessageBox.critical(self, "Config rejected", str(exc))

    def _save_campaign(self) -> None:
        try:
            validated_config_update(self.config_path, "campaign", "campaign.id", self.campaign_id.text().strip() or None)
            self._refresh_from_config(); self._log("Validated campaign ID saved.")
        except BaseException as exc:
            QtWidgets.QMessageBox.critical(self, "Campaign rejected", str(exc))

    def _approve_pilot(self) -> None:
        try:
            from .canonical_attempts import select_valid_attempt
            pilot = select_valid_attempt(self.cfg.output_root / str(self.cfg.campaign_id) / "pilot/run_1")
            stamp = datetime.now().astimezone().isoformat()
            validated_config_update(self.config_path, "safety", "approval.pilot_run_reference", str(pilot))
            validated_config_update(self.config_path, "safety", "approval.pilot_approved_at", stamp)
            validated_config_update(self.config_path, "safety", "approval.pilot_passed", True)
            self._refresh_from_config(); self._log(f"Human pilot PASS recorded: {pilot} at {stamp}")
        except BaseException as exc:
            QtWidgets.QMessageBox.critical(self, "Pilot approval rejected", str(exc))

    def _acknowledge_warmup(self) -> None:
        try:
            stamp = datetime.now().astimezone().isoformat()
            validated_config_update(self.config_path, "safety", "approval.warmup_acknowledged_at", stamp)
            self._refresh_from_config(); self._log(f"Warmup acknowledged at {stamp}")
        except BaseException as exc:
            QtWidgets.QMessageBox.critical(self, "Warmup acknowledgement rejected", str(exc))

    def _fail_pilot(self) -> None:
        try:
            validated_config_update(self.config_path, "safety", "approval.pilot_passed", False)
            validated_config_update(self.config_path, "safety", "approval.pilot_run_reference", None)
            validated_config_update(self.config_path, "safety", "approval.pilot_approved_at", None)
            self._refresh_from_config(); self._log("Human pilot FAIL recorded; canonical runs remain locked.")
        except BaseException as exc:
            QtWidgets.QMessageBox.critical(self, "Pilot FAIL record rejected", str(exc))

    def _edit_config(self) -> None:
        choices = sorted(f"{role}:{field}" for role, fields in EDITABLE_FIELDS.items() for field in fields)
        selected, ok = QtWidgets.QInputDialog.getItem(self, "Canonical config editor", "Validated field", choices, 0, False)
        if not ok: return
        raw, ok = QtWidgets.QInputDialog.getText(self, "Canonical config editor", "YAML scalar value (null allowed)")
        if not ok: return
        import yaml
        try:
            role, field = selected.split(":", 1)
            value = yaml.safe_load(raw)
            validated_config_update(self.config_path, role, field, value)
            self._refresh_from_config(); self._log(f"Validated config saved: {role}:{field}")
        except BaseException as exc:
            QtWidgets.QMessageBox.critical(self, "Config rejected; original retained", str(exc))

    def _mode_key(self) -> str:
        return ("manual", "pilot", "static", "delay", "dynamic")[self.mode.currentIndex()]

    def _relative(self, backend_mode: str) -> str:
        mechanical, repeat = self.mechanical.currentText(), int(self.repeat.currentText())
        if backend_mode == "manual": return datetime.now().strftime("%Y%m%d_%H%M%S_manual")
        if backend_mode == "pilot": return "pilot/run_1"
        if backend_mode == "static": return f"static/{mechanical}/{self.approach.currentText()}/repeat_{repeat}"
        if backend_mode == "delay": return "delay/probe_1"
        return f"dynamic/{mechanical}/{self.trajectory.currentText()}/repeat_{repeat}"

    def refresh_progress(self) -> None:
        if not hasattr(self, "progress_table"): return
        remaining = cooldown_remaining_sec(self.cfg) if self.cfg.campaign_id else 0.0
        if not self.active_run:
            self.run_button.setEnabled(remaining <= 0.0)
            if remaining > 0: self.run_button.setText(f"Cooldown remaining: {remaining:.1f} s")
            else: self.run_button.setText("RUN + SAVE")
        mode = self._mode_key()
        if mode not in ("static", "dynamic") or self.cfg.campaign_id is None:
            self.progress_table.setRowCount(0); self.progress_summary.setText("No canonical matrix for this mode")
            self.next_run.setText("-"); return
        rows = progress_rows(self.cfg, mode)
        headers = ("Mechanical", "Trajectory/Approach", "Repeat", "State", "Attempts", "Invalid", "Selected")
        self.progress_table.setColumnCount(len(headers)); self.progress_table.setHorizontalHeaderLabels(headers)
        self.progress_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            condition = row.get("approach_direction") or row["trajectory"]
            values = (row["mechanical_configuration"], condition, row["repeat"], row["state"], row["attempt_count"], row["invalid_attempt_count"], row["selected_attempt"] or "-")
            for j, value in enumerate(values): self.progress_table.setItem(i, j, QtWidgets.QTableWidgetItem(str(value)))
        valid = sum(row["state"] == "VALID" for row in rows)
        invalid = sum(int(row["invalid_attempt_count"]) for row in rows)
        next_row = next((row for row in rows if row["state"] != "VALID"), None)
        self.progress_summary.setText(f"Completed/Valid: {valid}/{len(rows)} | Invalid attempts: {invalid} | Remaining: {len(rows)-valid}")
        self.next_run.setText(next_row["relative_directory"] if next_row else "COMPLETE")

    def _selected_progress_row(self) -> dict[str, Any]:
        mode = self._mode_key()
        if mode not in ("static", "dynamic"):
            raise ValueError("Static 또는 Dynamic progress row를 먼저 선택하십시오.")
        index = self.progress_table.currentRow()
        rows = progress_rows(self.cfg, mode)
        if index < 0 or index >= len(rows):
            raise ValueError("progress table에서 logical run을 선택하십시오.")
        return rows[index]

    def _prepare_retry(self) -> None:
        """Load one immutable logical run into the controls; RUN creates a new attempt."""
        try:
            row = self._selected_progress_row()
            self.mechanical.setCurrentText(str(row["mechanical_configuration"]))
            self.repeat.setCurrentText(str(row["repeat"]))
            if self._mode_key() == "static":
                self.approach.setCurrentText(str(row["approach_direction"]))
            else:
                self.trajectory.setCurrentText(str(row["trajectory"]))
            self.override.setChecked(True)
            self.override_reason.setText("explicit immutable retry selected in GUI")
            self._log(f"Retry prepared: {row['relative_directory']}. RUN confirmation will create a new attempt; no overwrite.")
        except BaseException as exc:
            QtWidgets.QMessageBox.warning(self, "Retry unavailable", str(exc))

    def _select_accepted_attempt(self) -> None:
        try:
            from .canonical_attempts import write_attempt_selection
            row = self._selected_progress_row()
            choices = list(row["valid_attempts"])
            if not choices:
                raise ValueError("이 logical run에는 valid attempt가 없습니다.")
            selected, ok = QtWidgets.QInputDialog.getItem(
                self, "Select accepted attempt", "Valid immutable attempt", choices, 0, False
            )
            if not ok:
                return
            logical = self.cfg.output_root / str(self.cfg.campaign_id) / str(row["relative_directory"])
            pointer = write_attempt_selection(logical, str(selected))
            self.refresh_progress()
            self._log(f"Accepted attempt selected explicitly: {pointer} -> {selected}")
        except BaseException as exc:
            QtWidgets.QMessageBox.warning(self, "Attempt selection rejected", str(exc))

    def preview(self) -> None:
        mode = self._mode_key()
        self.plots["position"].clear()
        if mode == "manual":
            duration, rate = self.manual_duration.value(), 100.0
            t = [i/rate for i in range(round(duration*rate))]
            if self.manual_target.currentText() == "Hold": q = [self.manual_center.value() for _ in t]
            elif self.manual_target.currentText() == "Step": q = [self.manual_center.value() + (self.manual_amplitude.value() if x >= duration/2 else 0.0) for x in t]
            else: q = [self.manual_center.value()+self.manual_amplitude.value()*math.sin(2*math.pi*self.manual_frequency.value()*x) for x in t]
            summary = f"MANUAL TEST — NOT PART OF CANONICAL DATASET | {len(t)} samples | {duration:.3f} s"
        elif mode == "pilot":
            from .canonical_trajectories import build_pilot
            samples = build_pilot(self.cfg)
            t = [s.scheduled_time_sec for s in samples]; q = [s.goal_position_rad for s in samples]
            summary = f"pilot: {len(samples)} target updates at 50 Hz; bus preview repeats each target at 100 Hz"
        else:
            preview = build_preview(self.cfg, mode, self.trajectory.currentText(), self.approach.currentText())
            t = [s.scheduled_time_sec for s in preview.samples]; q = [s.goal_position_rad for s in preview.samples]
            summary = f"{mode}: {len(t)} samples, {preview.duration_sec:.3f} s, range [{preview.minimum_rad:.5f},{preview.maximum_rad:.5f}] rad, max discrete speed {preview.maximum_discrete_speed_rad_s:.5f} rad/s"
            if mode == "static":
                previous = None
                for sample in preview.samples:
                    if sample.phase != previous:
                        line = pg.InfiniteLine(sample.scheduled_time_sec, angle=90, pen=pg.mkPen((120, 120, 120, 100)))
                        self.plots["position"].addItem(line)
                        previous = sample.phase
        self.plots["position"].plot(t, q, pen=pg.mkPen("y", width=2), name="Goal")
        self._log(summary + "\nPreview did not open the motor port.")

    def run(self) -> None:
        mode = self._mode_key()
        backend_mode = {"pilot": "pilot", "static": "static", "delay": "delay", "dynamic": "collect", "manual": "manual"}[mode]
        mechanical = self.mechanical.currentText()
        configured_mechanical = (
            self.cfg.pilot.get("mechanical_configuration") if mode == "pilot" else
            self.cfg.trajectories["delay_probe"].get("mechanical_configuration") if mode == "delay" else
            mechanical
        )
        if mode in ("pilot", "delay") and mechanical != configured_mechanical:
            QtWidgets.QMessageBox.warning(
                self, "Canonical setup mismatch",
                f"{mode} mechanical configuration은 {configured_mechanical}입니다. Setup selection을 맞추십시오."
            )
            return
        load = self.cfg.configuration(mechanical).load
        measured = self.cfg.loads[load].get("measured_mass_kg")
        length = self.cfg.arm_length_m(mechanical)
        text = (f"PHYSICAL SETUP CONFIRMATION\n\nMechanical: {mechanical}\nArm: {length*1000:.0f} mm\n"
                f"Load: {load}, measured={measured} kg\nMode 5 / P={self.cfg.registers.get('position_p_gain')} / D={self.cfg.registers.get('position_d_gain')}\n"
                f"Campaign: {self.cfg.campaign_id}\n\n실제 시험대가 위 상태와 일치합니까?")
        if QtWidgets.QMessageBox.question(self, "CONFIRM & RUN", text) != QtWidgets.QMessageBox.Yes:
            return
        if self.override.isChecked() and not self.override_reason.text().strip():
            QtWidgets.QMessageBox.warning(self, "Override rejected", "Override reason is required."); return
        confirmed = {"mechanical_configuration": mechanical, "arm_length_m": length,
                     "load": load, "measured_mass_kg": measured,
                     "confirmed_at": datetime.now().astimezone().isoformat()}
        command = {"action": "run", "config": str(self.config_path), "mode": backend_mode,
                   "confirm": ("MANUAL_MX106_MODE5" if backend_mode == "manual" else CONFIRMATIONS[backend_mode]), "physical_setup_confirmation": confirmed,
                   "mechanical_configuration": mechanical, "trajectory": self.trajectory.currentText() if mode == "dynamic" else ("static_calibration" if mode == "static" else ("delay_probe" if mode == "delay" else ("pilot" if mode == "pilot" else f"manual_{self.manual_target.currentText().lower()}"))),
                   "approach": self.approach.currentText(), "repeat": int(self.repeat.currentText()),
                   "relative": self._relative(backend_mode), "override_reason": self.override_reason.text().strip() if self.override.isChecked() else "",
                   "duration_sec": self.manual_duration.value(), "manual_target_type": self.manual_target.currentText(),
                   "keep_temporary_log": self.keep_manual_log.isChecked(),
                   "manual_center_rad": self.manual_center.value(), "manual_amplitude_rad": self.manual_amplitude.value(),
                   "manual_frequency_hz": self.manual_frequency.value()}
        self.active_run = True; self.inbox.put(command)

    def _analysis_action(self, action: str) -> None:
        value = self.result_dir.text().strip()
        if not value:
            QtWidgets.QMessageBox.warning(self, "Result required", "result directory를 입력하십시오.")
            return
        path = Path(value)
        if not path.is_absolute(): path = self.cfg.project_root / path
        self.inbox.put({"action": action, "config": str(self.config_path), "result_dir": str(path)})

    def request_torque_off(self) -> None:
        self.inbox.put({"action": "torque_off", "config": str(self.config_path)})
        self._log("TORQUE OFF requested. This software action does not replace physical emergency power cutoff.")

    def _drain_worker(self) -> None:
        while True:
            try: message = self.outbox.get_nowait()
            except queue.Empty: break
            kind = message.get("type")
            if kind == "telemetry": self._telemetry(message)
            else:
                self._log(str(message))
                if kind == "connected":
                    self.connected = True
                    readback = message.get("readback", {})
                    self.register_status.setText("Connected; canonical readback captured")
                    self.register_table.setRowCount(len(readback))
                    for row, (name, value) in enumerate(sorted(readback.items())):
                        self.register_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(name)))
                        self.register_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(value)))
                if kind == "disconnected": self.connected = False
                if kind in ("completed", "aborted", "error"): self.active_run = False; self.refresh_progress()

    def _telemetry(self, row: dict[str, Any]) -> None:
        mapping = {"t": "host_time_sec", "goal": "goal_position_rad", "q": "present_position_rad", "qd": "present_velocity_rad_s", "current": "present_current_A", "pwm": "present_pwm_fraction"}
        for key, field in mapping.items(): self.buffers[key].append(float(row[field]))
        for key in self.buffers: self.buffers[key] = self.buffers[key][-3000:]
        t = self.buffers["t"]
        self.plots["position"].clear(); self.plots["position"].plot(t, self.buffers["goal"], pen="y"); self.plots["position"].plot(t, self.buffers["q"], pen="c")
        for plot, key in (("current", "current"), ("velocity", "qd"), ("pwm", "pwm")):
            self.plots[plot].clear(); self.plots[plot].plot(t, self.buffers[key], pen="c")
        rate = 0.0
        if len(t) >= 2 and t[-1] > t[-2]: rate = 1.0 / (t[-1] - t[-2])
        self.live_numbers.setText(
            f"Voltage {float(row.get('input_voltage_V', 0)):.2f} V | Temperature {float(row.get('temperature_C', 0)):.1f} C | "
            f"Current sat {int(float(row.get('current_saturated', 0)))} | PWM sat {int(float(row.get('pwm_saturated', 0)))} | "
            f"Telemetry {rate:.1f} Hz"
        )
        self.live_state.setText(
            f"Goal {float(row.get('goal_position_rad', 0)):+.5f} rad | "
            f"Position {float(row.get('present_position_rad', 0)):+.5f} rad | "
            f"Velocity {float(row.get('present_velocity_rad_s', 0)):+.5f} rad/s | "
            f"Current {float(row.get('present_current_A', 0)):+.4f} A | "
            f"PWM {float(row.get('present_pwm_fraction', 0)):+.4f} | "
            f"Tick {row.get('realtime_tick_raw', '—')} | Moving {row.get('moving', '—')} | "
            f"Status {row.get('moving_status', '—')}"
        )

    def _log(self, text: str) -> None:
        self.status.appendPlainText(text)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.active_run:
            self.request_torque_off()
        self.inbox.put({"action": "shutdown", "config": str(self.config_path)})
        self.worker.join(timeout=2.0)
        if self.worker.is_alive(): self.worker.terminate(); self.worker.join(timeout=1.0)
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe desktop GUI for canonical MX-106 Mode-5 experiments")
    parser.add_argument("--config", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--mock", action="store_true", help="explicit hardware-free mock; never falls back automatically")
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(args.config, args.mock); window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
