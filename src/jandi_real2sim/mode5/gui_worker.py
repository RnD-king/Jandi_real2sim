"""Process-isolated GUI worker.  Mock mode is explicit and never a fallback."""

from __future__ import annotations

import math
import queue
import shutil
import time
from pathlib import Path
from multiprocessing.queues import Queue
from typing import Any

from .canonical_acquisition import collect_run
from .canonical_bus import CanonicalMode5Bus
from .canonical_config import load_canonical_campaign
from .canonical_trajectories import Sample, build_delay, build_dynamic, build_pilot, build_static
from .gui_backend import completed_run_summary, progress_rows, require_physical_confirmation
from .spec import CONFIRMATIONS


def _emit(outbox: Queue, kind: str, **payload: Any) -> None:
    outbox.put({"type": kind, **payload})


def _mock_run(outbox: Queue, control: Queue, command: dict[str, Any]) -> None:
    duration = float(command.get("duration_sec", 3.0))
    rate = 100.0
    started = time.monotonic()
    for index in range(max(1, round(duration * rate))):
        try:
            pending = control.get_nowait()
        except queue.Empty:
            pending = None
        if pending and pending.get("action") in ("torque_off", "shutdown"):
            _emit(outbox, "aborted", reason="operator_abort", mock=True)
            return
        t = index / rate
        q_goal = 0.2 * math.sin(2 * math.pi * 0.5 * t)
        q = 0.19 * math.sin(2 * math.pi * 0.5 * max(0.0, t - 0.02))
        _emit(outbox, "telemetry", host_time_sec=t, goal_position_rad=q_goal,
              present_position_rad=q, present_velocity_rad_s=0.6 * math.cos(math.pi * t),
              present_current_A=0.25 * (q_goal - q), present_pwm_fraction=0.08,
              input_voltage_V=12.0, temperature_C=31, current_saturated=0,
              pwm_saturated=0, mock=True)
        deadline = started + (index + 1) / rate
        time.sleep(max(0.0, deadline - time.monotonic()))
    _emit(outbox, "completed", mock=True, saved=False,
          note="Explicit mock telemetry is never written to the canonical raw dataset.")


def _samples(cfg, mode: str, command: dict[str, Any]):
    if mode == "static":
        return build_static(cfg, str(command["approach"]))
    if mode == "delay":
        return build_delay(cfg)
    if mode == "collect":
        return build_dynamic(cfg, str(command["trajectory"]))
    if mode == "pilot":
        return build_pilot(cfg)
    if mode == "manual":
        rate = cfg.command_rate_hz
        duration = float(command["duration_sec"])
        center = float(command["manual_center_rad"])
        amplitude = float(command["manual_amplitude_rad"])
        frequency = float(command["manual_frequency_hz"])
        target_type = str(command["manual_target_type"])
        count = max(1, round(duration * rate))
        values = []
        for index in range(count):
            t = index / rate
            if target_type == "Hold": goal = center
            elif target_type == "Step": goal = center + (amplitude if t >= duration / 2 else 0.0)
            elif target_type == "Sine": goal = center + amplitude * math.sin(2 * math.pi * frequency * t)
            else: raise ValueError(f"unknown manual target type: {target_type}")
            cfg.rad_to_raw(goal)
            values.append(Sample(index, t, f"manual_{target_type.lower()}", goal))
        return values
    raise ValueError(f"worker canonical mode가 아닙니다: {mode}")


def _real_run(outbox: Queue, control: Queue, command: dict[str, Any]) -> None:
    cfg = load_canonical_campaign(command["config"])
    mode = str(command["mode"])
    expected = "MANUAL_MX106_MODE5" if mode == "manual" else CONFIRMATIONS[mode]
    if command.get("confirm") != expected:
        raise PermissionError(f"canonical confirmation mismatch: expected {expected}")
    physical = require_physical_confirmation(command.get("physical_setup_confirmation"))
    relative = str(command["relative"])
    if mode in ("static", "collect"):
        rows = progress_rows(cfg, mode)
        next_row = next((row for row in rows if row["state"] != "VALID"), None)
        if next_row and next_row["relative_directory"] != relative and not str(command.get("override_reason", "")).strip():
            raise PermissionError(f"planned NEXT RUN은 {next_row['relative_directory']}입니다.")

    def abort_requested() -> bool:
        try:
            pending = control.get_nowait()
        except queue.Empty:
            return False
        return pending.get("action") in ("torque_off", "shutdown")

    def telemetry(row: dict[str, object]) -> None:
        _emit(outbox, "telemetry", **row)

    target = collect_run(
        cfg, mode, relative, str(command["mechanical_configuration"]),
        str(command["trajectory"]), int(command["repeat"]), _samples(cfg, mode, command),
        str(command.get("override_reason") or "") or None,
        physical_setup_confirmation=physical, telemetry_callback=telemetry,
        abort_requested=abort_requested,
        per_run_override=(
            {"target_type": command["manual_target_type"], "center_rad": command["manual_center_rad"],
             "amplitude_rad": command["manual_amplitude_rad"], "frequency_hz": command["manual_frequency_hz"],
             "duration_sec": command["duration_sec"]}
            if mode == "manual" else None
        ),
    )
    summary = completed_run_summary(target, cfg)
    if mode == "delay":
        try:
            from .canonical_analysis import _load_run, estimate_delay
            summary["delay_calibration"] = estimate_delay(cfg, _load_run(target))
        except BaseException as exc:
            summary["delay_diagnostic_error"] = repr(exc)
    saved = True
    emitted_path = str(target)
    if mode == "manual" and not bool(command.get("keep_temporary_log", True)):
        manual_root = (cfg.project_root / "data/temp/manual").resolve()
        resolved_target = target.resolve()
        if not resolved_target.is_relative_to(manual_root):
            raise RuntimeError(f"manual temporary cleanup path가 허용 root 밖입니다: {resolved_target}")
        shutil.rmtree(resolved_target)
        saved = False
        emitted_path = ""
    _emit(outbox, "completed", path=emitted_path, mock=False, saved=saved, summary=summary)


def worker_main(inbox: Queue, outbox: Queue, *, mock: bool = False) -> None:
    """Long-lived process entry. Merely starting it never opens a serial port."""
    _emit(outbox, "worker_ready", mock=mock)
    while True:
        command = inbox.get()
        action = command.get("action")
        try:
            if action == "shutdown":
                _emit(outbox, "worker_stopped")
                return
            if action == "preview":
                _emit(outbox, "preview_ack")
            elif action == "connect":
                if mock:
                    _emit(outbox, "connected", mock=True, readback={"operating_mode": 5})
                else:
                    cfg = load_canonical_campaign(command["config"])
                    with CanonicalMode5Bus(cfg) as bus:
                        model = bus.ping()
                        readback = bus.read_configuration_snapshot()
                        state = bus.read_state()
                    _emit(outbox, "connected", mock=False, model_number=model,
                          readback=readback, state=state.__dict__)
            elif action == "disconnect":
                _emit(outbox, "disconnected")
            elif action == "torque_off":
                if not mock:
                    cfg = load_canonical_campaign(command["config"])
                    with CanonicalMode5Bus(cfg) as bus:
                        bus.torque(False)
                _emit(outbox, "torque_off_complete", mock=mock)
            elif action == "run":
                _emit(outbox, "run_started", mock=mock)
                if mock:
                    _mock_run(outbox, inbox, command)
                else:
                    _real_run(outbox, inbox, command)
            elif action in ("fit", "validate", "report"):
                cfg = load_canonical_campaign(command["config"])
                from .canonical_analysis import fit, report, validate
                if action == "fit":
                    path = fit(cfg, cfg.project_root / "configs/mode5/fit.yaml")
                elif action == "validate":
                    path = validate(cfg, Path(command["result_dir"]))
                else:
                    path = report(cfg, Path(command["result_dir"]))
                _emit(outbox, "analysis_completed", action=action, path=str(path))
            else:
                raise ValueError(f"unknown worker action: {action}")
        except BaseException as exc:
            _emit(outbox, "error", error=repr(exc), action=action, mock=mock)
