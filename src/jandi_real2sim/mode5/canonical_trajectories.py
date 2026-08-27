"""Canonical pilot/static/delay/dynamic command generation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .canonical_config import CanonicalCampaign
from .spec import APPROACH_DIRECTIONS, MAIN_TRAJECTORIES, REPEATS


@dataclass(frozen=True)
class Sample:
    sample_index: int
    scheduled_time_sec: float
    phase: str
    goal_position_rad: float


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    mechanical_configuration: str
    trajectory: str
    repeat: int
    approach_direction: str | None = None

    @property
    def relative_directory(self) -> str:
        if self.experiment == "static":
            assert self.approach_direction is not None
            return f"static/{self.mechanical_configuration}/{self.approach_direction}/repeat_{self.repeat}"
        return f"dynamic/{self.mechanical_configuration}/{self.trajectory}/repeat_{self.repeat}"


def static_run_specs(cfg: CanonicalCampaign) -> tuple[RunSpec, ...]:
    specs = [
        RunSpec("static", item.id, "static_calibration", repeat, approach)
        for item in cfg.configurations
        for approach in APPROACH_DIRECTIONS
        for repeat in REPEATS
    ]
    if cfg.execution_order == "randomized":
        random.Random(int(cfg.randomization_seed)).shuffle(specs)
    elif cfg.execution_order not in (None, "grouped"):
        raise ValueError("execution_order는 grouped 또는 randomized여야 합니다.")
    return tuple(specs)


def dynamic_run_specs(cfg: CanonicalCampaign) -> tuple[RunSpec, ...]:
    specs = [
        RunSpec("dynamic", item.id, trajectory, repeat)
        for item in cfg.configurations
        for trajectory in MAIN_TRAJECTORIES
        for repeat in REPEATS
    ]
    if cfg.execution_order == "randomized":
        random.Random(int(cfg.randomization_seed)).shuffle(specs)
    elif cfg.execution_order not in (None, "grouped"):
        raise ValueError("execution_order는 grouped 또는 randomized여야 합니다.")
    return tuple(specs)


def _append(samples: list[Sample], goal: float, duration: float, rate: float, phase: str) -> None:
    count = max(1, round(duration * rate))
    for _ in range(count):
        index = len(samples)
        samples.append(Sample(index, index / rate, phase, goal))


def _sample(samples: list[Sample], goal: float, rate: float, phase: str) -> None:
    index = len(samples)
    samples.append(Sample(index, index / rate, phase, goal))


def _smooth(samples: list[Sample], start: float, target: float, duration: float, rate: float, phase: str) -> None:
    """Append a half-cosine transfer without an adjacent-sample step."""
    if duration <= 0:
        raise ValueError(f"{phase} duration은 양수여야 합니다.")
    count = max(2, math.ceil(duration * rate))
    for local in range(1, count + 1):
        ratio = local / count
        blend = 0.5 - 0.5 * math.cos(math.pi * ratio)
        _sample(samples, start + blend * (target - start), rate, phase)


def _validate_waveform(cfg: CanonicalCampaign, samples: list[Sample], maximum_speed: float | None) -> None:
    if not samples:
        raise ValueError("빈 trajectory는 허용되지 않습니다.")
    for sample in samples:
        cfg.rad_to_raw(sample.goal_position_rad)
    if maximum_speed is not None:
        if maximum_speed <= 0:
            raise ValueError("maximum_command_speed_rad_s는 양수여야 합니다.")
        maximum_actual = max(
            (abs(right.goal_position_rad - left.goal_position_rad) * cfg.command_rate_hz
             for left, right in zip(samples, samples[1:])),
            default=0.0,
        )
        if maximum_actual > maximum_speed + 1e-12:
            raise ValueError(
                f"generated command speed {maximum_actual:.6g} rad/s가 "
                f"configured limit {maximum_speed:.6g} rad/s를 초과합니다."
            )


def command_events(samples: list[Sample]) -> tuple[Sample, ...]:
    """Return the ZOH command changes from a sampled command plan."""
    if not samples:
        return ()
    events = [samples[0]]
    for sample in samples[1:]:
        if sample.goal_position_rad != events[-1].goal_position_rad:
            events.append(sample)
    return tuple(events)


def build_pilot(cfg: CanonicalCampaign) -> list[Sample]:
    center = float(cfg.pilot["center_rad"])
    amplitude = float(cfg.pilot["amplitude_rad"])
    hold = float(cfg.pilot["hold_sec"])
    if amplitude <= 0 or hold <= 0:
        raise ValueError("pilot amplitude/hold는 양수여야 합니다.")
    result: list[Sample] = []
    for phase, goal in (
        ("center", center), ("positive", center + amplitude),
        ("center_after_positive", center), ("negative", center - amplitude),
        ("center_after_negative", center),
    ):
        cfg.rad_to_raw(goal)
        _append(result, goal, hold, cfg.command_rate_hz, phase)
    return result


def build_static(cfg: CanonicalCampaign, approach: str) -> list[Sample]:
    spec = cfg.trajectories["static_calibration"]
    angles = [float(value) for value in spec["static_angles_rad"]]
    offset = float(spec["approach_offset_rad"])
    transition = float(spec["approach_duration_sec"])
    transfer = float(spec["inter_point_transfer_duration_sec"])
    settle = float(spec["fixed_settling_hold_sec"])
    average = float(spec["averaging_window_sec"])
    if not angles or offset <= 0 or min(transition, transfer, settle, average, float(spec["minimum_settling_sec"])) <= 0:
        raise ValueError("static angles는 비어 있지 않고 모든 시간/offset은 양수여야 합니다.")
    sign = -1.0 if approach == "approach_positive" else 1.0
    ordered = sorted(angles, reverse=approach == "approach_negative")
    result: list[Sample] = []
    rate = cfg.command_rate_hz
    for point, target in enumerate(ordered):
        start = target + sign * offset
        cfg.rad_to_raw(start)
        cfg.rad_to_raw(target)
        if result:
            _smooth(result, result[-1].goal_position_rad, start, transfer, rate, f"point_{point}_inter_point_transfer")
        _append(result, start, float(spec["minimum_settling_sec"]), rate, f"point_{point}_approach_start")
        _smooth(result, start, target, transition, rate, f"point_{point}_approach")
        _append(result, target, settle, rate, f"point_{point}_settling")
        _append(result, target, average, rate, f"point_{point}_averaging")
    _validate_waveform(cfg, result, float(spec["maximum_command_speed_rad_s"]))
    return result


def build_delay(cfg: CanonicalCampaign) -> list[Sample]:
    spec = cfg.trajectories["delay_probe"]
    center = float(spec["center_rad"])
    hold = float(spec["hold_sec"])
    repeats = int(spec["repeats"])
    amplitudes = [float(value) for value in spec["step_amplitudes_rad"]]
    if not amplitudes or any(value <= 0 for value in amplitudes) or hold <= 0 or repeats < 1:
        raise ValueError("delay amplitudes/hold/repeats가 유효하지 않습니다.")
    result: list[Sample] = []
    _append(result, center, hold, cfg.command_rate_hz, "baseline")
    for repeat in range(1, repeats + 1):
        for amplitude_index, amplitude in enumerate(amplitudes):
            for direction, sign in (("positive", 1.0), ("negative", -1.0)):
                target = center + sign * amplitude
                cfg.rad_to_raw(target)
                _append(result, target, hold, cfg.command_rate_hz, f"r{repeat}_a{amplitude_index}_{direction}")
                _append(result, center, hold, cfg.command_rate_hz, f"r{repeat}_a{amplitude_index}_center")
    return result


def build_dynamic(cfg: CanonicalCampaign, name: str) -> list[Sample]:
    spec = cfg.trajectories[name]
    rate = cfg.command_rate_hz
    center = float(spec["center_rad"])
    result: list[Sample] = []
    center_hold = float(spec["center_hold_sec"])
    if center_hold <= 0:
        raise ValueError("center_hold_sec은 양수여야 합니다.")
    _append(result, center, center_hold, rate, "center")
    if name == "accelerated_oscillation":
        duration = float(spec["duration_sec"])
        f0 = float(spec["start_frequency_hz"])
        f1 = float(spec["end_frequency_hz"])
        amplitude = float(spec["amplitude_rad"])
        if amplitude <= 0 or duration <= 0 or not 0 < f0 <= f1 < rate / 2:
            raise ValueError("accelerated oscillation amplitude/duration/frequency가 유효하지 않습니다.")
        count = max(2, round(duration * rate))
        chirp_rate = (f1 - f0) / duration
        for local in range(count):
            t = local / rate
            phase = 2.0 * math.pi * (f0 * t + 0.5 * chirp_rate * t * t)
            _sample(result, center + amplitude * math.sin(phase), rate, "accelerated_oscillation")
    elif name == "slow_plus_highfreq":
        duration = float(spec["duration_sec"])
        if duration <= 0 or float(spec["slow_amplitude_rad"]) <= 0 or float(spec["high_frequency_amplitude_rad"]) <= 0:
            raise ValueError("slow+high-frequency duration/amplitudes는 양수여야 합니다.")
        if not 0 < float(spec["slow_frequency_hz"]) < float(spec["high_frequency_hz"]) < rate / 2:
            raise ValueError("slow/high frequencies의 순서 또는 Nyquist 범위가 유효하지 않습니다.")
        count = max(2, round(duration * rate))
        for local in range(count):
            t = local / rate
            goal = center
            goal += float(spec["slow_amplitude_rad"]) * math.sin(2 * math.pi * float(spec["slow_frequency_hz"]) * t)
            goal += float(spec["high_frequency_amplitude_rad"]) * math.sin(2 * math.pi * float(spec["high_frequency_hz"]) * t)
            _sample(result, goal, rate, "slow_plus_highfreq")
    elif name == "slowly_raise_lower":
        low, high = float(spec["lower_rad"]), float(spec["upper_rad"])
        speed, cycles = float(spec["speed_rad_s"]), int(spec["cycles"])
        endpoint_hold = float(spec["endpoint_hold_sec"])
        if not low < high or speed <= 0 or cycles < 1 or endpoint_hold <= 0:
            raise ValueError("slowly raise/lower 범위·속도·cycle·hold가 유효하지 않습니다.")
        transition = float(spec["transition_duration_sec"])
        _smooth(result, center, low, transition, rate, "transition_center_to_lower")
        for cycle in range(cycles):
            for label, start, target in (("raise", low, high), ("lower", high, low)):
                count = max(2, math.ceil(abs(target - start) / speed * rate) + 1)
                for local in range(count):
                    ratio = local / (count - 1)
                    _sample(result, start + ratio * (target - start), rate, f"cycle_{cycle}_{label}")
                _append(result, target, endpoint_hold, rate, f"cycle_{cycle}_{label}_hold")
    else:
        raise ValueError(f"canonical main trajectory가 아닙니다: {name}")
    if result[-1].goal_position_rad != center:
        _smooth(result, result[-1].goal_position_rad, center, float(spec["transition_duration_sec"]), rate, "transition_to_center")
    _append(result, center, center_hold, rate, "final_center")
    _validate_waveform(cfg, result, float(spec["maximum_command_speed_rad_s"]))
    return result
