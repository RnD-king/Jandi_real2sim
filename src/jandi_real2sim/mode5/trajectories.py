from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Mode5Campaign


@dataclass(frozen=True)
class Sample:
    cycle_index: int
    time_s: float
    phase: str
    goal_rad: float


def _hold(samples: list[Sample], goal: float, seconds: float, rate: int, phase: str) -> None:
    count = round(seconds * rate)
    if count < 1:
        raise ValueError(f"{phase}: hold 시간은 한 표본 이상이어야 합니다.")
    for _ in range(count):
        cycle = len(samples)
        samples.append(Sample(cycle, cycle / rate, phase, goal))


def step(cfg: Mode5Campaign) -> list[Sample]:
    spec = cfg.trajectories["step"]
    hold_sec = float(spec["hold_sec"])
    amplitudes = tuple(float(value) for value in spec["amplitudes_rad"])
    if not amplitudes or any(value <= 0 for value in amplitudes):
        raise ValueError("step amplitudes_rad는 하나 이상의 양수여야 합니다.")
    samples: list[Sample] = []
    _hold(samples, cfg.hardware.center_rad, hold_sec, cfg.timing.command_rate_hz, "center")
    for index, amplitude in enumerate(amplitudes, start=1):
        for side, sign in (("plus", 1.0), ("minus", -1.0)):
            goal = cfg.hardware.center_rad + sign * amplitude
            cfg.validate_motion(goal)
            _hold(samples, goal, hold_sec, cfg.timing.command_rate_hz, f"{side}_a{index}")
            _hold(
                samples,
                cfg.hardware.center_rad,
                hold_sec,
                cfg.timing.command_rate_hz,
                f"center_after_{side}_a{index}",
            )
    return samples


def pilot_step(cfg: Mode5Campaign) -> list[Sample]:
    """본 campaign과 분리된 center,+small,center,-small,center 안전 pilot."""
    hold_sec = float(cfg.trajectories["step"]["hold_sec"])
    amplitude = cfg.safety.pilot_amplitude_rad
    samples: list[Sample] = []
    for phase, delta in (
        ("center", 0.0),
        ("plus_pilot", amplitude),
        ("center_after_plus", 0.0),
        ("minus_pilot", -amplitude),
        ("center_after_minus", 0.0),
    ):
        goal = cfg.hardware.center_rad + delta
        cfg.validate_motion(goal)
        _hold(samples, goal, hold_sec, cfg.timing.command_rate_hz, phase)
    return samples


def triangle(cfg: Mode5Campaign) -> list[Sample]:
    spec = cfg.trajectories["triangle"]
    amplitude = float(spec["amplitude_rad"])
    cycles_each = int(spec["cycles_each"])
    frequencies = tuple(float(value) for value in spec["frequencies_hz"])
    if amplitude <= 0 or cycles_each < 1 or not frequencies:
        raise ValueError("triangle 설정이 유효하지 않습니다.")
    samples: list[Sample] = []
    rate = cfg.timing.command_rate_hz
    _hold(samples, cfg.hardware.center_rad, float(spec["center_hold_sec"]), rate, "center")
    for freq_index, frequency in enumerate(frequencies, start=1):
        if frequency <= 0 or frequency >= rate / 2:
            raise ValueError("triangle frequency가 유효하지 않습니다.")
        count = round(cycles_each / frequency * rate)
        start = len(samples)
        for local in range(count + 1):
            t = local / rate
            wave = (2.0 / math.pi) * math.asin(math.sin(2 * math.pi * frequency * t))
            goal = cfg.hardware.center_rad + amplitude * wave
            cfg.validate_motion(goal)
            cycle = len(samples)
            samples.append(Sample(cycle, cycle / rate, f"triangle_f{freq_index}", goal))
        _hold(samples, cfg.hardware.center_rad, float(spec["center_hold_sec"]), rate, f"center_after_f{freq_index}")
        if len(samples) <= start:
            raise AssertionError("triangle 표본 생성 실패")
    return samples


def sine(cfg: Mode5Campaign) -> list[Sample]:
    spec = cfg.trajectories["sine"]
    rate = cfg.timing.command_rate_hz
    cycles_each = int(spec["cycles_each"])
    points = tuple(spec["points"])
    samples: list[Sample] = []
    _hold(samples, cfg.hardware.center_rad, float(spec["center_hold_sec"]), rate, "center")
    for index, point in enumerate(points, start=1):
        frequency = float(point["frequency_hz"])
        amplitude = float(point["amplitude_rad"])
        if frequency <= 0 or frequency >= rate / 2 or amplitude <= 0 or cycles_each < 1:
            raise ValueError("sine point가 유효하지 않습니다.")
        count = round(cycles_each / frequency * rate)
        for local in range(count + 1):
            t = local / rate
            goal = cfg.hardware.center_rad + amplitude * math.sin(2 * math.pi * frequency * t)
            cfg.validate_motion(goal)
            cycle = len(samples)
            samples.append(Sample(cycle, cycle / rate, f"sine_f{index}", goal))
        _hold(samples, cfg.hardware.center_rad, float(spec["center_hold_sec"]), rate, f"center_after_f{index}")
    return samples


def build(cfg: Mode5Campaign, trajectory: str) -> list[Sample]:
    builders = {"step": step, "triangle": triangle, "sine": sine}
    try:
        samples = builders[trajectory](cfg)
    except KeyError as exc:
        raise ValueError(f"trajectory는 {tuple(builders)} 중 하나여야 합니다.") from exc
    if any(sample.cycle_index != index for index, sample in enumerate(samples)):
        raise AssertionError("trajectory cycle_index가 연속적이지 않습니다.")
    return samples
