"""BAM-compatible command trajectories plus no-load probes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Campaign, LOADED_TRAJECTORIES, NO_LOAD_TRAJECTORIES


@dataclass(frozen=True)
class Sample:
    index: int
    time_sec: float
    phase: str
    goal_rad: float
    torque_enable: bool = True


def _add(result: list[Sample], cfg: Campaign, duration: float, phase: str,
         function, torque_enable: bool = True) -> None:
    count = max(1, round(duration * cfg.command_rate_hz))
    for local in range(count):
        t = local / cfg.command_rate_hz
        goal = float(function(t, duration))
        cfg.rad_to_raw(goal)
        result.append(Sample(len(result), len(result) / cfg.command_rate_hz,
                             phase, goal, torque_enable))


def _smooth(a: float, b: float, t: float, duration: float) -> float:
    ratio = min(1.0, max(0.0, t / duration))
    return a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * ratio))


def build(cfg: Campaign, name: str) -> tuple[Sample, ...]:
    if name not in LOADED_TRAJECTORIES + NO_LOAD_TRAJECTORIES:
        raise KeyError(name)
    spec = cfg.trajectories[name]
    center = float(spec["center_rad"])
    result: list[Sample] = []
    if name == "delay_probe":
        hold = float(spec["hold_sec"])
        _add(result, cfg, hold, "baseline", lambda *_: center)
        for repeat in range(int(spec["repeats"])):
            for amplitude in map(float, spec["step_amplitudes_rad"]):
                for label, sign in (("positive", 1.0), ("negative", -1.0)):
                    _add(result, cfg, hold, f"r{repeat + 1}_{label}_{amplitude:g}",
                         lambda _t, _d, q=center + sign * amplitude: q)
                    _add(result, cfg, hold, f"r{repeat + 1}_center", lambda *_: center)
    elif name == "backlash_probe":
        frequency = float(spec["frequency_hz"])
        amplitude = float(spec["amplitude_rad"])
        duration = int(spec["cycles"]) / frequency
        _add(result, cfg, duration, "slow_triangle", lambda t, _d:
             center + amplitude * (2.0 / math.pi) * math.asin(math.sin(2.0 * math.pi * frequency * t)))
    elif name == "sin_time_square":
        duration = float(spec["duration_sec"]); f0 = float(spec["start_frequency_hz"])
        f1 = float(spec["end_frequency_hz"]); amplitude = float(spec["amplitude_rad"])
        chirp = (f1 - f0) / duration
        _add(result, cfg, duration, name, lambda t, _d:
             center + amplitude * math.sin(2.0 * math.pi * (f0 * t + 0.5 * chirp * t * t)))
    elif name == "sin_sin":
        duration = float(spec["duration_sec"])
        _add(result, cfg, duration, name, lambda t, _d:
             center + float(spec["slow_amplitude_rad"]) * math.sin(2.0 * math.pi * float(spec["slow_frequency_hz"]) * t)
             + float(spec["high_amplitude_rad"]) * math.sin(2.0 * math.pi * float(spec["high_frequency_hz"]) * t))
    elif name == "up_and_down":
        upper, returned = float(spec["upper_rad"]), float(spec["return_rad"])
        up, down = float(spec["raise_duration_sec"]), float(spec["lower_duration_sec"])
        _add(result, cfg, up, "raise", lambda t, d: _smooth(center, upper, t, d))
        _add(result, cfg, down, "lower", lambda t, d: _smooth(upper, returned, t, d))
        _add(result, cfg, float(spec["final_hold_sec"]), "hold", lambda *_: returned)
    elif name == "lift_and_drop":
        target = float(spec["lift_rad"]); duration = float(spec["lift_duration_sec"])
        _add(result, cfg, duration, "lift", lambda t, d: _smooth(center, target, t, d))
        _add(result, cfg, float(spec["release_duration_sec"]), "released",
             lambda *_: target, torque_enable=False)
        # Acquisition may end the release early on an angle/velocity threshold.
        # These recovery samples are replaced by a measured-state interpolation
        # at runtime, so re-enabling torque never commands a stale lift angle.
        _add(result, cfg, float(spec["recovery_duration_sec"]), "recovery",
             lambda t, d: _smooth(target, center, t, d), torque_enable=True)
        _add(result, cfg, float(spec["recovery_hold_sec"]), "recovery_hold",
             lambda *_: center, torque_enable=True)
    return tuple(result)


def trajectories_for(condition_id: str) -> tuple[str, ...]:
    return NO_LOAD_TRAJECTORIES if condition_id == "no_load" else LOADED_TRAJECTORIES
