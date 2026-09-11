"""Isolated hardware/optimizer worker for the Mode-3 GUI."""

from __future__ import annotations

import math
import queue
import time
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

from .acquisition import collect
from .analysis import compare_models, fit_backlash, fit_model, fit_time_controller
from .bus import Mode3Bus
from .calibrate import (
    SIGN_CONFIRMATION, _cfg_for, apply_readback, calibration_report,
    capture_upright_zero, discover, infer_signs, jog_raw_positive,
    readback_device, save_calibration_report, verify_sign_test_configuration,
)
from .config import load_campaign
from .trajectories import trajectories_for


def _emit(outbox: Queue, kind: str, **values: Any) -> None:
    outbox.put({"type": kind, **values})


def worker_main(inbox: Queue, outbox: Queue, mock: bool = False) -> None:
    calibration: dict[str, Any] = {}
    _emit(outbox, "ready", mock=mock)
    while True:
        command = inbox.get(); action = command.get("action")
        try:
            if action == "shutdown": return
            config_path = Path(command["config"])
            if action == "calibration_discover":
                if mock:
                    device = {"serial_device": "/dev/ttyUSB_MOCK", "baudrate": 3_000_000,
                              "motor_id": 0, "model_number": 321, "firmware_version": 45}
                    snapshot = {
                        "firmware_version": 45, "return_delay_time_raw": 250,
                        "homing_offset_raw": 0, "drive_mode": 0, "operating_mode": 3,
                        "temperature_limit_c": 80, "max_voltage_limit_raw": 160,
                        "min_voltage_limit_raw": 100, "pwm_limit_raw": 885,
                        "current_limit_raw": 2047, "velocity_limit_raw": 1023,
                        "max_position_limit_raw": 4095, "min_position_limit_raw": 0,
                        "position_d_gain": 0, "position_i_gain": 0, "position_p_gain": 850,
                        "feedforward_2nd_gain": 0, "feedforward_1st_gain": 0,
                        "profile_acceleration": 0, "profile_velocity": 0,
                        "status_return_level_raw": 2, "bus_watchdog_raw": 0,
                        "hardware_error": 0, "torque_enable": 0,
                    }
                    present = {"position_raw": 2048, "input_voltage_v": 12.0, "temperature_c": 30}
                    report_path = "[mock: not saved]"
                else:
                    device = discover(command.get("port"), command.get("baudrate"), command.get("motor_id"))
                    snapshot, present = readback_device(device)
                    report = calibration_report(device, snapshot, present, None, None)
                    report_path = str(save_calibration_report(config_path, report))
                calibration.clear()
                calibration.update(device=device, snapshot=snapshot, present=present,
                                   zero_raw=None, currents=None, pwms=None, signs=None)
                _emit(outbox, "calibration_discovered", device=device, snapshot=snapshot,
                      present=present, report_path=report_path, mock=mock)
            elif action == "calibration_capture_zero":
                if not calibration.get("device"):
                    raise RuntimeError("먼저 Auto discover / readback을 실행하십시오.")
                if mock:
                    zero_raw = 2048
                else:
                    with Mode3Bus(_cfg_for(calibration["device"])) as bus:
                        verify_sign_test_configuration(bus)
                        zero_raw = capture_upright_zero(bus)
                calibration.update(zero_raw=zero_raw, currents=None, pwms=None, signs=None)
                _emit(outbox, "calibration_zero_captured", zero_raw=zero_raw)
            elif action == "calibration_jog":
                if command.get("confirm") != SIGN_CONFIRMATION:
                    raise PermissionError("Mode-3 sign-test confirmation mismatch")
                if calibration.get("zero_raw") is None:
                    raise RuntimeError("먼저 도립 q=0을 저장하십시오.")
                jog_ticks = int(command.get("jog_ticks", 32))
                if mock:
                    currents, pwms = [8] * jog_ticks, [25] * jog_ticks
                    time.sleep(.1)
                else:
                    with Mode3Bus(_cfg_for(calibration["device"])) as bus:
                        currents, pwms = jog_raw_positive(bus, int(calibration["zero_raw"]), jog_ticks)
                calibration.update(currents=currents, pwms=pwms, signs=None)
                _emit(outbox, "calibration_jog_completed", jog_ticks=jog_ticks)
            elif action == "calibration_direction":
                if calibration.get("currents") is None or calibration.get("pwms") is None:
                    raise RuntimeError("먼저 방향 시험을 실행하십시오.")
                signs = infer_signs(bool(command["raw_positive_is_joint_positive"]),
                                    calibration["currents"], calibration["pwms"])
                calibration["signs"] = signs
                _emit(outbox, "calibration_signs_inferred", signs=signs)
            elif action == "calibration_apply":
                required = ("device", "snapshot", "present", "zero_raw", "signs")
                missing = [name for name in required if calibration.get(name) is None]
                if missing:
                    raise RuntimeError(f"캘리브레이션 단계가 완료되지 않았습니다: {missing}")
                report = calibration_report(calibration["device"], calibration["snapshot"],
                                            calibration["present"], calibration["zero_raw"],
                                            calibration["signs"])
                if mock:
                    report_path = "[mock: not saved]"
                else:
                    apply_readback(config_path.resolve(), calibration["device"], calibration["snapshot"],
                                   calibration["zero_raw"], calibration["signs"])
                    report_path = str(save_calibration_report(config_path, report))
                _emit(outbox, "calibration_applied", report_path=report_path, mock=mock)
            elif action == "connect":
                cfg = load_campaign(command["config"])
                if mock: _emit(outbox, "connected", readback={"operating_mode": 3, "position_p_gain": 850}, mock=True)
                else:
                    cfg = load_campaign(command["config"], require_hardware=True)
                    with Mode3Bus(cfg) as bus:
                        model = bus.ping(); readback = bus.read_configuration_snapshot()
                    _emit(outbox, "connected", model=model, readback=readback, mock=False)
            elif action == "torque_off":
                cfg = load_campaign(command["config"])
                if not mock:
                    cfg = load_campaign(command["config"], require_hardware=True)
                    with Mode3Bus(cfg) as bus: bus.torque(False)
                _emit(outbox, "torque_off")
            elif action == "collect_drop_pilot":
                cfg = load_campaign(command["config"], require_hardware=not mock,
                                    require_bench=True)
                if command.get("confirm") != cfg.confirmations["collect"]:
                    raise PermissionError("Mode-3 BAM hardware confirmation mismatch")
                condition = "mass1_distance1"
                _emit(outbox, "run_started", trajectory="drop_safety_pilot")
                if mock:
                    for i in range(60):
                        t = i / 100; goal = -.1 if t < .3 else 0.0
                        _emit(outbox, "telemetry", host_time_sec=t, goal_position_rad=goal,
                              present_position_rad=max(-.4, -.1 - .8 * max(0, t-.2)),
                              present_velocity_rad_s=-2.0 if .2 <= t < .35 else 0.0,
                              present_pwm_fraction=.1, present_current_A=.2,
                              input_voltage_V=12., temperature_C=30.)
                        time.sleep(.001)
                    path = "[mock: not saved]"
                else:
                    def aborted() -> bool:
                        try: pending = inbox.get_nowait()
                        except queue.Empty: return False
                        return pending.get("action") in ("torque_off", "shutdown")
                    path = str(collect(
                        cfg, condition, "lift_and_drop", 1,
                        telemetry=lambda row: _emit(outbox, "telemetry", **row),
                        abort=aborted, run_group="pilot/drop_safety", role_override="pilot",
                    ))
                _emit(outbox, "drop_pilot_completed", condition=condition, path=path)
            elif action == "collect_condition":
                cfg = load_campaign(command["config"])
                condition = str(command["condition"]); repeat = int(command["repeat"])
                cfg = load_campaign(command["config"], require_hardware=not mock,
                                    require_bench=condition != "no_load")
                if command.get("confirm") != cfg.confirmations["collect"]:
                    raise PermissionError("Mode-3 BAM hardware confirmation mismatch")
                _emit(outbox, "batch_started", condition=condition, repeat=repeat)
                for trajectory in trajectories_for(condition):
                    # Every trajectory restarts its local clock at zero. Tell
                    # the GUI to break the previous plot before new samples.
                    _emit(outbox, "run_started", trajectory=trajectory)
                    if mock:
                        for i in range(100):
                            t = i / 100; goal = .1 * math.sin(2 * math.pi * t)
                            _emit(outbox, "telemetry", host_time_sec=t, goal_position_rad=goal,
                                  present_position_rad=.095 * math.sin(2 * math.pi * max(0, t-.01)),
                                  present_velocity_rad_s=.5 * math.cos(2*math.pi*t), present_pwm_fraction=.1,
                                  present_current_A=.2, input_voltage_V=12., temperature_C=30.)
                            time.sleep(.001)
                        _emit(outbox, "run_completed", trajectory=trajectory, path="[mock: not saved]")
                    else:
                        def aborted() -> bool:
                            try: pending = inbox.get_nowait()
                            except queue.Empty: return False
                            return pending.get("action") in ("torque_off", "shutdown")
                        path = collect(cfg, condition, trajectory, repeat,
                                       telemetry=lambda row: _emit(outbox, "telemetry", **row), abort=aborted)
                        _emit(outbox, "run_completed", trajectory=trajectory, path=str(path))
                        time.sleep(float(cfg.safety["between_runs_sec"]))
                _emit(outbox, "batch_completed", condition=condition, repeat=repeat)
            elif action == "fit_time":
                cfg = load_campaign(command["config"])
                _emit(outbox, "analysis_completed", stage=action, path=str(fit_time_controller(cfg)))
            elif action in ("fit_m1", "fit_m2", "fit_m3", "fit_m4", "fit_m5"):
                cfg = load_campaign(command["config"])
                model = action[-2:]
                _emit(outbox, "analysis_completed", stage=action, path=str(fit_model(cfg, model)))
            elif action == "compare":
                cfg = load_campaign(command["config"])
                _emit(outbox, "analysis_completed", stage=action, path=str(compare_models(cfg)))
            elif action == "fit_backlash":
                cfg = load_campaign(command["config"])
                _emit(outbox, "analysis_completed", stage=action, path=str(fit_backlash(cfg)))
            else: raise KeyError(action)
        except BaseException as exc:
            _emit(outbox, "error", action=action, error=repr(exc))
