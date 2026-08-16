from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectorySample:
    cycle_index: int
    time_s: float
    phase: str
    q_cmd_rad: dict[str, float]


def _sample_count(duration_s: float, rate_hz: int) -> int:
    if duration_s <= 0.0:
        raise ValueError("duration_s는 양수여야 합니다.")
    return max(1, round(duration_s * rate_hz))


def smooth_transition(
    start: Mapping[str, float],
    target: Mapping[str, float],
    duration_s: float,
    rate_hz: int = 100,
    *,
    cycle_offset: int = 0,
    phase: str = "transition",
) -> Iterator[TrajectorySample]:
    if tuple(start) != tuple(target):
        raise ValueError("start와 target의 관절 순서가 다릅니다.")
    count = _sample_count(duration_s, rate_hz)
    for index in range(count + 1):
        fraction = index / count
        blend = 0.5 - 0.5 * math.cos(math.pi * fraction)
        command = {
            name: float(start[name]) + blend * (float(target[name]) - float(start[name]))
            for name in start
        }
        yield TrajectorySample(
            cycle_index=cycle_offset + index,
            time_s=(cycle_offset + index) / rate_hz,
            phase=phase,
            q_cmd_rad=command,
        )


def compact_joint_step(
    center: Mapping[str, float],
    joint_name: str,
    amplitude_rad: float,
    hold_sec: float,
    repeats: int = 1,
    rate_hz: int = 100,
) -> Iterator[TrajectorySample]:
    if joint_name not in center:
        raise KeyError(joint_name)
    if amplitude_rad <= 0.0:
        raise ValueError("amplitude_rad는 양수여야 합니다.")
    if repeats < 1:
        raise ValueError("repeats는 1 이상이어야 합니다.")
    count = _sample_count(hold_sec, rate_hz)
    cycle = 0
    levels = (("center", 0.0), ("plus", amplitude_rad), ("center", 0.0),
              ("minus", -amplitude_rad), ("center", 0.0))
    for repeat in range(repeats):
        for phase_name, delta in levels:
            command = dict(center)
            command[joint_name] += delta
            for _ in range(count):
                yield TrajectorySample(
                    cycle_index=cycle,
                    time_s=cycle / rate_hz,
                    phase=f"repeat{repeat + 1}_{phase_name}",
                    q_cmd_rad=dict(command),
                )
                cycle += 1


def hold_pose(
    pose: Mapping[str, float],
    duration_s: float,
    rate_hz: int = 100,
    *,
    phase: str = "hold",
) -> Iterator[TrajectorySample]:
    """고정된 12관절 자세를 duration_s 동안 유지한다."""
    count = _sample_count(duration_s, rate_hz)
    for cycle in range(count):
        yield TrajectorySample(
            cycle_index=cycle,
            time_s=cycle / rate_hz,
            phase=phase,
            q_cmd_rad=dict(pose),
        )


def compact_joint_steps(
    center: Mapping[str, float],
    joint_name: str,
    amplitudes_rad: tuple[float, ...],
    hold_sec: float,
    rate_hz: int = 100,
) -> Iterator[TrajectorySample]:
    """center,+A,center,-A,center를 진폭마다 이어 붙인 식별용 step이다."""
    if joint_name not in center:
        raise KeyError(joint_name)
    if not amplitudes_rad or any(amplitude <= 0.0 for amplitude in amplitudes_rad):
        raise ValueError("amplitudes_rad는 하나 이상의 양수여야 합니다.")
    if any(b <= a for a, b in zip(amplitudes_rad, amplitudes_rad[1:])):
        raise ValueError("amplitudes_rad는 작은 값부터 엄격히 증가해야 합니다.")
    count = _sample_count(hold_sec, rate_hz)
    levels: list[tuple[str, float]] = [("center", 0.0)]
    for amplitude_index, amplitude in enumerate(amplitudes_rad, start=1):
        label = f"a{amplitude_index}"
        levels.extend(
            (
                (f"plus_{label}", amplitude),
                (f"center_after_plus_{label}", 0.0),
                (f"minus_{label}", -amplitude),
                (f"center_after_minus_{label}", 0.0),
            )
        )
    cycle = 0
    for phase_name, delta in levels:
        command = dict(center)
        command[joint_name] += delta
        for _ in range(count):
            yield TrajectorySample(
                cycle_index=cycle,
                time_s=cycle / rate_hz,
                phase=phase_name,
                q_cmd_rad=dict(command),
            )
            cycle += 1


def triangle_joint(
    center: Mapping[str, float],
    joint_name: str,
    amplitude_rad: float,
    frequency_hz: float,
    cycles: int,
    rate_hz: int = 100,
) -> Iterator[TrajectorySample]:
    """방향전환의 마찰·백래시를 보기 위한 0 시작/종료 대칭 triangle 입력이다."""
    if joint_name not in center:
        raise KeyError(joint_name)
    if amplitude_rad <= 0.0 or frequency_hz <= 0.0 or cycles < 1:
        raise ValueError("진폭·주파수는 양수이고 cycles는 1 이상이어야 합니다.")
    duration_s = cycles / frequency_hz
    count = _sample_count(duration_s, rate_hz)
    for cycle in range(count + 1):
        time_s = cycle / rate_hz
        # asin(sin())은 [-1,1] triangle이며 t=0과 정수 주기 끝에서 0이다.
        wave = (2.0 / math.pi) * math.asin(
            math.sin(2.0 * math.pi * frequency_hz * time_s)
        )
        command = dict(center)
        command[joint_name] += amplitude_rad * wave
        yield TrajectorySample(
            cycle_index=cycle,
            time_s=time_s,
            phase="triangle",
            q_cmd_rad=command,
        )


def multisine_joint(
    center: Mapping[str, float],
    joint_name: str,
    amplitude_rad: float,
    frequencies_hz: tuple[float, ...],
    duration_s: float,
    seed: int,
    rate_hz: int = 100,
    *,
    fade_sec: float = 1.0,
) -> Iterator[TrajectorySample]:
    """동일 진폭 배분·무작위 위상·cosine fade를 쓰는 재현 가능한 multisine이다."""
    if joint_name not in center:
        raise KeyError(joint_name)
    if amplitude_rad <= 0.0 or duration_s <= 0.0:
        raise ValueError("진폭과 duration_s는 양수여야 합니다.")
    if not frequencies_hz or any(frequency <= 0.0 for frequency in frequencies_hz):
        raise ValueError("frequencies_hz는 하나 이상의 양수여야 합니다.")
    if len(set(frequencies_hz)) != len(frequencies_hz):
        raise ValueError("multisine 주파수는 중복될 수 없습니다.")
    if max(frequencies_hz) >= rate_hz / 2.0:
        raise ValueError("multisine 주파수는 Nyquist 주파수보다 작아야 합니다.")
    if fade_sec < 0.0 or 2.0 * fade_sec > duration_s:
        raise ValueError("fade_sec는 0 이상이며 duration_s의 절반 이하여야 합니다.")

    count = _sample_count(duration_s, rate_hz)
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=len(frequencies_hz))
    component_amplitude = amplitude_rad / len(frequencies_hz)
    fade_count = round(fade_sec * rate_hz)
    for cycle in range(count):
        time_s = cycle / rate_hz
        signal = sum(
            component_amplitude
            * math.sin(2.0 * math.pi * frequency * time_s + phase)
            for frequency, phase in zip(frequencies_hz, phases, strict=True)
        )
        envelope = 1.0
        if fade_count:
            if cycle < fade_count:
                envelope = 0.5 - 0.5 * math.cos(math.pi * cycle / fade_count)
            elif cycle >= count - fade_count:
                remaining = count - 1 - cycle
                envelope = 0.5 - 0.5 * math.cos(
                    math.pi * max(0, remaining) / fade_count
                )
        command = dict(center)
        command[joint_name] += envelope * signal
        yield TrajectorySample(
            cycle_index=cycle,
            time_s=time_s,
            phase="multisine",
            q_cmd_rad=command,
        )
