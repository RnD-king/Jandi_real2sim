"""Seven-condition desktop tool for Mode-3 BAM acquisition and ordered fitting."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import sys
from pathlib import Path

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .config import (
    DEFAULT_CAMPAIGN, DISTANCE_KEYS, MASS_KEYS, load_campaign,
    trajectory_profile_matches, update_value,
)
from .gui_worker import worker_main
from .trajectories import trajectories_for


def _drop_pilot_valid(cfg) -> bool:
    if not cfg.campaign_id:
        return False
    root = (cfg.output_root / str(cfg.campaign_id) / "pilot" / "drop_safety" /
            "mass1_distance1" / "lift_and_drop" / "repeat_1")
    for path in root.glob("attempt_*"):
        metadata = path / "metadata.json"
        if not metadata.exists():
            continue
        report = json.loads(metadata.read_text())
        if (report.get("valid") and
                trajectory_profile_matches(cfg, "lift_and_drop", report) and
                report.get("drop_catch", {}).get("reason") == "angle" and
                "drop_emergency" not in report):
            return True
    return False


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: Path, mock: bool = False):
        super().__init__(); self.config_path = config.resolve(); self.mock = mock
        self.cfg = load_campaign(self.config_path); self.inbox: mp.Queue = mp.Queue(); self.outbox: mp.Queue = mp.Queue()
        self.worker = mp.Process(target=worker_main, args=(self.inbox, self.outbox, mock), daemon=True); self.worker.start()
        self.active = False; self.analysis_active = False; self.analysis_action = None
        self.buffers = {key: [] for key in ("t", "goal", "q", "dq", "pwm", "current")}
        self.setWindowTitle("MX-106 Mode 3 · BAM identification" + (" [MOCK]" if mock else "")); self.resize(1450, 900)
        self._build(); self.refresh(); self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self._drain); self.timer.start(30)

    def _build(self) -> None:
        root = QtWidgets.QWidget(); self.setCentralWidget(root); outer = QtWidgets.QHBoxLayout(root)
        left = QtWidgets.QVBoxLayout(); outer.addLayout(left, 0); tabs = QtWidgets.QTabWidget(); outer.addWidget(tabs, 1)
        connection = QtWidgets.QGroupBox("Connection — Mode 3 / P=850"); form = QtWidgets.QFormLayout(connection)
        self.hardware = QtWidgets.QLabel(); self.hardware.setWordWrap(True); form.addRow(self.hardware)
        row = QtWidgets.QHBoxLayout(); self.connect = QtWidgets.QPushButton("Connect / Readback"); self.off = QtWidgets.QPushButton("TORQUE OFF")
        self.off.setStyleSheet("font-weight:bold;color:white;background:#b00020;min-height:42px")
        row.addWidget(self.connect); row.addWidget(self.off); form.addRow(row); left.addWidget(connection)
        self.connect.clicked.connect(lambda: self.inbox.put({"action":"connect", "config":str(self.config_path)}))
        self.off.clicked.connect(lambda: self.inbox.put({"action":"torque_off", "config":str(self.config_path)}))

        setup = QtWidgets.QGroupBox("Campaign / measured bench constants"); grid = QtWidgets.QGridLayout(setup)
        self.campaign = QtWidgets.QLineEdit(); self.repeat = QtWidgets.QComboBox(); self.repeat.addItems(("1", "2", "3"))
        grid.addWidget(QtWidgets.QLabel("Campaign ID"),0,0); grid.addWidget(self.campaign,0,1); grid.addWidget(QtWidgets.QLabel("Repeat"),0,2); grid.addWidget(self.repeat,0,3)
        self.fields = {}
        labels = (
            ("disk_mass_kg", "One disk mass [kg]"),
            ("disk_diameter_m", "Disk diameter [m]"),
            ("load_fastener_mass_kg", "Load fastener mass [kg]"),
            ("distance1", "Distance 1 [m]"), ("distance2", "Distance 2 [m]"),
            ("arm_mass_kg", "Arm mass [kg]"), ("arm_com_radius_m", "Arm COM [m]"),
            ("arm_inertia_about_com_kg_m2", "Arm COM inertia [kg m²]"),
            ("arm_inertia_kg_m2", "Arm pivot inertia [kg m²]"),
            ("no_load_horn_mass_kg", "No-load horn mass [kg]"),
            ("loaded_horn_mass_kg", "Loaded horn mass [kg]"),
            ("loaded_horn_fastener_mass_kg", "Loaded horn fasteners [kg]"),
        )
        for index, (key, label) in enumerate(labels, start=1):
            box = QtWidgets.QDoubleSpinBox(); box.setDecimals(7); box.setRange(0, 100); box.setSpecialValueText("unset")
            self.fields[key] = box; grid.addWidget(QtWidgets.QLabel(label), index, 0); grid.addWidget(box, index, 1, 1, 3)
        self.derived_loads = QtWidgets.QLabel(); self.derived_loads.setWordWrap(True)
        grid.addWidget(self.derived_loads, len(labels)+1, 0, 1, 4)
        self.save_button = QtWidgets.QPushButton("Validate & save campaign/bench values"); self.save_button.clicked.connect(self._save); grid.addWidget(self.save_button, len(labels)+2,0,1,4)
        left.addWidget(setup)

        pilot = QtWidgets.QGroupBox("Required before loaded campaign"); pilot_layout = QtWidgets.QVBoxLayout(pilot)
        self.drop_pilot_status = QtWidgets.QLabel(); self.drop_pilot_status.setWordWrap(True)
        self.drop_pilot_button = QtWidgets.QPushButton("Drop safety pilot — Mass1 × Distance1")
        self.drop_pilot_button.setMinimumHeight(42); self.drop_pilot_button.clicked.connect(self.run_drop_pilot)
        pilot_layout.addWidget(self.drop_pilot_status); pilot_layout.addWidget(self.drop_pilot_button)
        left.addWidget(pilot)

        conditions = QtWidgets.QGroupBox("One button = every trajectory for this condition"); cg = QtWidgets.QGridLayout(conditions)
        self.condition_buttons = {}
        entries = [("no_load", "No load calibration")]
        entries += [(f"{mass}_{distance}", f"{mass.title()} × {distance.title()}") for mass in MASS_KEYS for distance in DISTANCE_KEYS]
        for index, (key, label) in enumerate(entries):
            button = QtWidgets.QPushButton(label); button.setMinimumHeight(38); button.clicked.connect(lambda _=False, c=key: self.run_condition(c))
            self.condition_buttons[key] = button; cg.addWidget(button, index//2, index%2)
        left.addWidget(conditions); left.addStretch(1)

        calibration = QtWidgets.QWidget(); tabs.addTab(calibration, "Calibration")
        cal = QtWidgets.QVBoxLayout(calibration)
        intro = QtWidgets.QLabel(
            "실험 전 1→4 순서로 진행합니다. 1번은 Torque OFF readback이고, "
            "실제 움직임은 3번 방향 시험에서만 발생합니다."
        )
        intro.setWordWrap(True); cal.addWidget(intro)
        device_box = QtWidgets.QGroupBox("Optional device filters — 비우면 자동 탐색")
        device_form = QtWidgets.QFormLayout(device_box)
        self.cal_port = QtWidgets.QLineEdit(); self.cal_port.setPlaceholderText("예: /dev/ttyUSB0")
        self.cal_baud = QtWidgets.QLineEdit(); self.cal_baud.setPlaceholderText("예: 3000000")
        self.cal_id = QtWidgets.QLineEdit(); self.cal_id.setPlaceholderText("예: 0")
        device_form.addRow("Port", self.cal_port); device_form.addRow("Baudrate", self.cal_baud)
        device_form.addRow("Motor ID", self.cal_id); cal.addWidget(device_box)
        self.calibration_buttons = {}
        calibration_actions = (
            ("discover", "1. Auto discover / readback", self._calibration_discover),
            ("zero", "2. Save upright q=0", self._calibration_zero),
            ("jog", "3. Direction sign test — +32 ticks (2.81°)", self._calibration_jog),
            ("apply", "4. Apply measured values to YAML", self._calibration_apply),
        )
        for key, label, callback in calibration_actions:
            button = QtWidgets.QPushButton(label); button.setMinimumHeight(45)
            button.clicked.connect(callback); self.calibration_buttons[key] = button; cal.addWidget(button)
        self.calibration_status = QtWidgets.QPlainTextEdit(); self.calibration_status.setReadOnly(True)
        self.calibration_status.setPlaceholderText("캘리브레이션 결과가 여기에 표시됩니다.")
        cal.addWidget(self.calibration_status, 1)
        warning = QtWidgets.QLabel(
            "자동으로 정하지 않는 값: 추/막대 질량·거리·관성, software safety limits, "
            "실험 각도 범위, Bus Watchdog. 이 값들은 실제 치구와 실험 설계에 맞춰 입력해야 합니다."
        )
        warning.setWordWrap(True); cal.addWidget(warning)

        live = QtWidgets.QWidget(); tabs.addTab(live, "Live"); lg = QtWidgets.QGridLayout(live); self.plots = {}
        for index, (key, title) in enumerate((("position","Goal / position"),("velocity","Velocity"),("pwm","Present PWM"),("current","Present current"))):
            plot = pg.PlotWidget(title=title); plot.showGrid(x=True,y=True,alpha=.25); lg.addWidget(plot,index//2,index%2); self.plots[key]=plot
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); lg.addWidget(self.log,2,0,1,2)

        progress = QtWidgets.QWidget(); tabs.addTab(progress, "7 × 3 progress"); pg_layout = QtWidgets.QVBoxLayout(progress)
        self.progress = QtWidgets.QTableWidget(7,4); self.progress.setHorizontalHeaderLabels(("Condition","Repeat 1 (fit)","Repeat 2 (fit)","Repeat 3 (validation)")); self.progress.horizontalHeader().setStretchLastSection(True)
        pg_layout.addWidget(self.progress)

        fit = QtWidgets.QWidget(); tabs.addTab(fit, "Ordered fitting"); fg = QtWidgets.QVBoxLayout(fit)
        stages = (("fit_time","1. Fit time / controller characteristics"),("fit_m1","2. Fit M1 — Coulomb + viscous"),
                  ("fit_m2","3. Fit M2 — M1 + Stribeck"),("fit_m3","4. Fit M3 — M1 + load dependence"),
                  ("fit_m4","5. Fit M4 — Stribeck + load dependence"),
                  ("fit_m5","6. Fit M5 — directional load dependence"),
                  ("compare","7. Compare M1–M5 / select"),
                  ("fit_backlash","8. Fit effective state backlash"))
        self.fit_stage_labels = dict(stages); self.fit_buttons = {}
        for action, label in stages:
            button=QtWidgets.QPushButton(label); button.setMinimumHeight(45); button.clicked.connect(lambda _=False,a=action:self._analyze(a)); fg.addWidget(button)
            self.fit_buttons[action] = button
        note=QtWidgets.QLabel("Hard contract: repeats 1·2 only fit. Repeat 3 is validation-only.\nEach stage checks that its prerequisite result exists."); note.setWordWrap(True); fg.addWidget(note)
        self.fit_status = QtWidgets.QLabel("Idle — select the next ordered fitting stage.")
        self.fit_status.setWordWrap(True); fg.addWidget(self.fit_status)
        self.fit_progress = QtWidgets.QProgressBar(); self.fit_progress.setRange(0, 100); self.fit_progress.setValue(0)
        fg.addWidget(self.fit_progress)
        self.fit_log = QtWidgets.QPlainTextEdit(); self.fit_log.setReadOnly(True)
        self.fit_log.setPlaceholderText("Fitting progress, elapsed time, RMSE, completion path, and errors appear here.")
        fg.addWidget(self.fit_log, 1)

    def _save(self) -> None:
        try:
            update_value(self.config_path,"campaign",self.campaign.text().strip() or None)
            for key, box in self.fields.items(): update_value(self.config_path,key,None if box.value()==0 else box.value())
            self.refresh(); self._log("Saved validated YAML values.")
        except BaseException as exc: QtWidgets.QMessageBox.critical(self,"Save rejected",str(exc))

    @staticmethod
    def _optional_int(field: QtWidgets.QLineEdit, name: str) -> int | None:
        text = field.text().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{name}은 정수여야 합니다: {text}") from exc

    def _calibration_discover(self) -> None:
        try:
            command = {
                "action": "calibration_discover", "config": str(self.config_path),
                "port": self.cal_port.text().strip() or None,
                "baudrate": self._optional_int(self.cal_baud, "Baudrate"),
                "motor_id": self._optional_int(self.cal_id, "Motor ID"),
            }
            self.calibration_status.setPlainText("장치 탐색 및 Torque-OFF readback 중…")
            self.inbox.put(command)
        except BaseException as exc:
            QtWidgets.QMessageBox.critical(self, "Discovery rejected", str(exc))

    def _calibration_zero(self) -> None:
        text = (
            "추를 제거하고 혼 또는 가벼운 막대만 장착하십시오.\n"
            "추를 달았을 때 하늘을 똑바로 보는 도립 위치를 q=0으로 맞추십시오.\n\n"
            "이 버튼은 Torque OFF 상태에서 현재 encoder tick만 저장하며 모터를 움직이지 않습니다."
        )
        if QtWidgets.QMessageBox.question(self, "Capture upright q=0", text) != QtWidgets.QMessageBox.Yes:
            return
        self.inbox.put({"action": "calibration_capture_zero", "config": str(self.config_path)})

    def _calibration_jog(self) -> None:
        text = (
            "주변 ±3°가 비어 있고 추와 무거운 막대가 제거됐는지 확인하십시오.\n"
            "모터는 저장한 0점에서 raw +32 tick(약 2.81°)으로 이동한 뒤 복귀하고 Torque OFF됩니다.\n\n"
            "비상 정지 수단을 준비했습니까?"
        )
        if QtWidgets.QMessageBox.question(self, "Authorize low-amplitude motion", text) != QtWidgets.QMessageBox.Yes:
            return
        self.inbox.put({"action": "calibration_jog", "config": str(self.config_path),
                        "jog_ticks": 32, "confirm": "MOVE_MX106_MODE3_CALIBRATION"})

    def _calibration_apply(self) -> None:
        text = (
            "확인한 장치 정보, 영점, direction/pwm_direction과 PWM 축을 따르는 "
            "current_direction 및 gravity_torque_sign을 canonical YAML에 기록합니다.\n"
            "추·거리·관성 및 software safety limits는 변경하지 않습니다. 계속할까요?"
        )
        if QtWidgets.QMessageBox.question(self, "Apply calibration", text) != QtWidgets.QMessageBox.Yes:
            return
        self.inbox.put({"action": "calibration_apply", "config": str(self.config_path)})

    def refresh(self) -> None:
        self.cfg=load_campaign(self.config_path); self.campaign.setText(str(self.cfg.campaign_id or ""))
        self.hardware.setText(f"port={self.cfg.hardware.get('serial_device')} | baud={self.cfg.hardware.get('baudrate')} | ID={self.cfg.hardware.get('motor_id')}\nMode=3 | P=850 | D={self.cfg.registers.get('position_d_gain')} | command/state=100 Hz")
        bench_values = {
            "disk_mass_kg": self.cfg.bench["load"].get("disk_mass_kg"),
            "disk_diameter_m": self.cfg.bench["load"].get("disk_diameter_m"),
            "load_fastener_mass_kg": self.cfg.bench["load"].get("fastener_mass_kg"),
            **{key: self.cfg.bench["axis_to_load_com_m"].get(key) for key in DISTANCE_KEYS},
            "arm_mass_kg": self.cfg.bench["arm"].get("mass_kg"),
            "arm_com_radius_m": self.cfg.bench["arm"].get("com_radius_m"),
            "arm_inertia_about_com_kg_m2": self.cfg.bench["arm"].get("inertia_about_com_kg_m2"),
            "arm_inertia_kg_m2": self.cfg.bench["arm"].get("inertia_about_pivot_kg_m2"),
            "no_load_horn_mass_kg": self.cfg.bench["no_load_hardware"].get("horn_mass_kg"),
            "loaded_horn_mass_kg": self.cfg.bench["loaded_mounting_hardware"].get("horn_mass_kg"),
            "loaded_horn_fastener_mass_kg":
                self.cfg.bench["loaded_mounting_hardware"].get("horn_fastener_mass_kg"),
        }
        for key, box in self.fields.items():
            value = bench_values[key]; box.setValue(0 if value is None else float(value))
        derived = [self.cfg.condition(f"{key}_distance1") for key in MASS_KEYS]
        self.derived_loads.setText(
            "Derived load assemblies: " + " | ".join(
                f"{item.disk_count} disk: {item.mass_kg:.4f} kg, "
                f"I_COM={item.intrinsic_inertia_kg_m2:.9f} kg m²" for item in derived
            )
        )
        pilot_valid = _drop_pilot_valid(self.cfg)
        self.drop_pilot_status.setText(
            "PASS — loaded runs unlocked" if pilot_valid else
            "PENDING — run after no-load repeats, before the first loaded run"
        )
        conditions=["no_load"]+[f"{m}_{d}" for m in MASS_KEYS for d in DISTANCE_KEYS]
        root=self.cfg.output_root/str(self.cfg.campaign_id)
        for row, condition in enumerate(conditions):
            self.progress.setItem(row,0,QtWidgets.QTableWidgetItem(condition))
            for repeat in (1,2,3):
                total=len(trajectories_for(condition)); valid=0
                for trajectory in trajectories_for(condition):
                    logical=root/condition/trajectory/f"repeat_{repeat}"
                    if any(
                        (p/"metadata.json").exists()
                        and (report := json.loads((p/"metadata.json").read_text())).get("valid")
                        and trajectory_profile_matches(self.cfg, trajectory, report)
                        for p in logical.glob("attempt_*")
                    ): valid+=1
                self.progress.setItem(row,repeat,QtWidgets.QTableWidgetItem(f"{valid}/{total}"))

    def run_drop_pilot(self) -> None:
        try:
            cfg = load_campaign(self.config_path, require_hardware=not self.mock, require_bench=True)
            condition = cfg.condition("mass1_distance1")
            text = (
                "Mass1 × Distance1 장치를 조립하십시오.\n"
                f"Load assembly: {condition.mass_kg:.4f} kg at {condition.distance_m:.3f} m\n\n"
                "동작: -50° 이동 → Torque OFF → -70°부터 속도 연속 감속 "
                "→ -88° 이내 정지 목표 → 0° 복귀\n"
                "-93°는 경고, -95°는 측정된 물리 충돌 한계입니다.\n"
                "Drop 구간에서는 속도로 중단하지 않습니다.\n\n"
                "낙하 방향 완충장치와 비상 TORQUE OFF를 준비했습니까?"
            )
            if QtWidgets.QMessageBox.question(self, "Authorize drop safety pilot", text) != QtWidgets.QMessageBox.Yes:
                return
            self.active=True; self._set_buttons(False)
            self.inbox.put({"action":"collect_drop_pilot", "config":str(self.config_path),
                            "confirm":cfg.confirmations["collect"]})
        except BaseException as exc:
            QtWidgets.QMessageBox.critical(self, "Pilot rejected", str(exc))

    def run_condition(self, condition: str) -> None:
        try:
            cfg=load_campaign(self.config_path,require_hardware=not self.mock,require_bench=condition!="no_load")
            if condition != "no_load" and not self.mock:
                if not _drop_pilot_valid(cfg):
                    raise RuntimeError("Drop safety pilot을 먼저 성공시켜야 부하 본실험을 실행할 수 있습니다.")
            c=cfg.condition(condition); repeat=int(self.repeat.currentText()); trajectories=", ".join(trajectories_for(condition))
            text=(f"Mode 3 / P=850\nCondition: {condition}\nDisks: {c.disk_count}\n"
                  f"Load assembly mass: {c.mass_kg:.4f} kg\n"
                  f"Load intrinsic inertia: {c.intrinsic_inertia_kg_m2:.9f} kg m²\n"
                  f"Axis-load COM: {c.distance_m:.3f} m\nRepeat: {repeat} "
                  f"({'FIT' if repeat<3 else 'VALIDATION ONLY'})\nRuns: {trajectories}\n\n"
                  "Physical setup과 낙하 안전장치를 확인했습니까?")
            if QtWidgets.QMessageBox.question(self,"Physical confirmation",text)!=QtWidgets.QMessageBox.Yes:return
            self.active=True; self._set_buttons(False); self.inbox.put({"action":"collect_condition","config":str(self.config_path),"condition":condition,"repeat":repeat,"confirm":cfg.confirmations["collect"]})
        except BaseException as exc: QtWidgets.QMessageBox.critical(self,"Run rejected",str(exc))

    def _analyze(self, action: str) -> None:
        if self.analysis_active:
            QtWidgets.QMessageBox.information(self, "Fitting in progress", "현재 피팅이 끝날 때까지 기다리십시오.")
            return
        self.analysis_active = True
        self.analysis_action = action
        self._set_analysis_buttons(False)
        label = self.fit_stage_labels[action]
        self.fit_buttons[action].setText(f"RUNNING — {label}")
        self.fit_status.setText(f"Running — {label}")
        self.fit_progress.setRange(0, 0)
        self._fit_log(f"START | {label}")
        self._log(f"Requested: {action}")
        self.inbox.put({"action":action,"config":str(self.config_path)})

    def _set_analysis_buttons(self, enabled: bool) -> None:
        for button in self.fit_buttons.values(): button.setEnabled(enabled)
        self._set_buttons(enabled)
        self.connect.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        for button in self.calibration_buttons.values(): button.setEnabled(enabled)

    def _finish_analysis(self, *, success: bool, message: str) -> None:
        self.analysis_active = False
        for action, button in self.fit_buttons.items():
            button.setText(self.fit_stage_labels[action])
        self.analysis_action = None
        self._set_analysis_buttons(True)
        self.fit_progress.setRange(0, 100)
        self.fit_progress.setValue(100 if success else 0)
        self.fit_status.setText(("Completed — " if success else "Failed — ") + message)

    def _set_buttons(self, enabled: bool) -> None:
        for button in self.condition_buttons.values(): button.setEnabled(enabled)
        self.drop_pilot_button.setEnabled(enabled)

    def _drain(self) -> None:
        while True:
            try: message=self.outbox.get_nowait()
            except queue.Empty: break
            kind=message.get("type")
            if kind=="telemetry": self._telemetry(message)
            elif kind=="analysis_started":
                self.fit_status.setText(f"Running — {message.get('label', message.get('stage'))}")
                self._fit_log(f"WORKER STARTED | {message.get('label', message.get('stage'))}")
            elif kind=="analysis_progress":
                evaluation=int(message.get("evaluation", 0)); total=int(message.get("total", 0))
                elapsed=float(message.get("elapsed_sec", 0.0)); current=float(message.get("current_rmse", float("nan")))
                best=float(message.get("best_rmse", float("nan")))
                if total > 0:
                    self.fit_progress.setRange(0, total); self.fit_progress.setValue(min(evaluation, total))
                status=(f"Running — {message.get('stage')} | evaluation {evaluation}/{total} | "
                        f"elapsed {elapsed:.1f} s | RMSE current={current:.6f}, best={best:.6f} rad")
                details = ""
                if "command_delay_sec" in message:
                    details = f" | delay={float(message['command_delay_sec'])*1000:.2f} ms"
                elif message.get("parameters"):
                    details = " | " + ", ".join(
                        f"{name}={float(value):.6g}"
                        for name, value in message["parameters"].items()
                    )
                self.fit_status.setText(status); self._fit_log(status + details)
            elif kind=="analysis_completed":
                stage=str(message.get("stage")); path=str(message.get("path"))
                self._fit_log(f"COMPLETED | {stage} | {path}")
                self._finish_analysis(success=True, message=f"{stage}\n{path}")
                self._log(str(message)); self.refresh()
            elif kind=="run_started":
                self._reset_live()
                self._log(str(message))
            elif kind=="calibration_jog_completed":
                self._calibration_line(f"방향 시험 완료: +{message['jog_ticks']} raw ticks, Torque OFF")
                box = QtWidgets.QMessageBox(self)
                box.setWindowTitle("Observed direction")
                box.setText("방금 raw tick 증가 방향이 실험에서 정의한 +관절 방향이었습니까?")
                box.setInformativeText("Yes=같은 방향, No=반대 방향, Cancel=결과 폐기")
                box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
                answer = box.exec()
                if answer in (QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No):
                    self.inbox.put({"action": "calibration_direction", "config": str(self.config_path),
                                    "raw_positive_is_joint_positive": answer == QtWidgets.QMessageBox.Yes})
                else:
                    self._calibration_line("사용자가 방향 판정을 취소했습니다. YAML 적용 전 시험을 다시 하십시오.")
            elif kind=="calibration_discovered":
                device=message["device"]; snapshot=message["snapshot"]; present=message["present"]
                self._calibration_line(
                    f"발견: {device['serial_device']} @ {device['baudrate']}, ID={device['motor_id']}, "
                    f"model={device['model_number']}, firmware={device['firmware_version']}\n"
                    f"Mode={snapshot['operating_mode']}, P/I/D={snapshot['position_p_gain']}/"
                    f"{snapshot['position_i_gain']}/{snapshot['position_d_gain']}, "
                    f"position={present['position_raw']} raw, voltage={present['input_voltage_v']:.1f} V, "
                    f"temperature={present['temperature_c']} °C\nreport: {message['report_path']}"
                )
            elif kind=="calibration_zero_captured":
                self._calibration_line(f"도립 q=0 저장: encoder_zero_raw={message['zero_raw']} (YAML 미반영)")
            elif kind=="calibration_signs_inferred":
                signs=message["signs"]
                self._calibration_line("부호 판정: " + ", ".join(f"{key}={value}" for key,value in signs.items()))
            elif kind=="calibration_applied":
                self._calibration_line(
                    ("MOCK: YAML은 변경하지 않았습니다." if message.get("mock") else "YAML 반영 완료.")
                    + f"\nreport: {message['report_path']}"
                )
                self.refresh()
            elif kind=="drop_pilot_completed":
                self._log(str(message))
                self.active=False; self._set_buttons(True); self.refresh()
                QtWidgets.QMessageBox.information(
                    self, "Drop safety pilot completed",
                    f"파일럿이 정상 완료되었습니다.\n{message['path']}\n\n"
                    "그래프와 drop_catch metadata를 확인한 뒤 부하 본실험을 진행하십시오.",
                )
            else:
                self._log(str(message))
                if kind=="error" and str(message.get("action")) in self.fit_stage_labels:
                    action=str(message.get("action")); error=str(message.get("error"))
                    self._fit_log(f"ERROR | {action} | {error}")
                    self._finish_analysis(success=False, message=f"{action}: {error}")
                if kind=="error" and str(message.get("action", "")).startswith("calibration_"):
                    self._calibration_line(f"ERROR [{message.get('action')}]: {message.get('error')}")
                    QtWidgets.QMessageBox.critical(self, "Calibration failed", str(message.get("error")))
                if kind in ("batch_completed","error"):
                    self.active=False; self._set_buttons(True); self.refresh()

    def _reset_live(self) -> None:
        self.buffers = {key: [] for key in ("t", "goal", "q", "dq", "pwm", "current")}
        for plot in self.plots.values():
            plot.clear()

    def _telemetry(self,row) -> None:
        mapping={"t":"host_time_sec","goal":"goal_position_rad","q":"present_position_rad","dq":"present_velocity_rad_s","pwm":"present_pwm_fraction","current":"present_current_A"}
        for key,field in mapping.items(): self.buffers[key].append(float(row[field])); self.buffers[key]=self.buffers[key][-3000:]
        t=self.buffers["t"]; self.plots["position"].clear(); self.plots["position"].plot(t,self.buffers["goal"],pen="y"); self.plots["position"].plot(t,self.buffers["q"],pen="c")
        for plot,key in (("velocity","dq"),("pwm","pwm"),("current","current")): self.plots[plot].clear(); self.plots[plot].plot(t,self.buffers[key],pen="c")

    def _log(self,text:str)->None:self.log.appendPlainText(text)
    def _fit_log(self,text:str)->None:self.fit_log.appendPlainText(text)
    def _calibration_line(self,text:str)->None:self.calibration_status.appendPlainText(text)
    def closeEvent(self,event:QtGui.QCloseEvent)->None:
        if self.active:self.inbox.put({"action":"torque_off","config":str(self.config_path)})
        self.inbox.put({"action":"shutdown","config":str(self.config_path)}); self.worker.join(timeout=2)
        if self.worker.is_alive():self.worker.terminate()
        event.accept()


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=DEFAULT_CAMPAIGN); parser.add_argument("--mock",action="store_true"); args=parser.parse_args()
    mp.set_start_method("spawn",force=True); app=QtWidgets.QApplication(sys.argv); window=MainWindow(args.config,args.mock); window.show(); raise SystemExit(app.exec())


if __name__=="__main__":main()
