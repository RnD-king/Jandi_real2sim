from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from jandi_real2sim.config import MUJOCO_DOF_ORDER


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]

    return {
        "min": ordered[0],
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _fmt(values: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in values.items())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jandi Real2Sim 실행 폴더의 구조·100 Hz·진단값 검증"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="구조·표본·Hardware Error·deadline 문제가 있으면 종료코드 1",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    metadata_path = run_dir / "metadata.json"
    telemetry_path = run_dir / "telemetry.csv"
    metadata = json.loads(metadata_path.read_text())
    rows = list(csv.DictReader(telemetry_path.open()))
    failures: list[str] = []

    expected = int(metadata["expected_sample_count"])
    if len(rows) != expected:
        failures.append(f"표본 수 actual={len(rows)}, expected={expected}")
    cycles = [int(row["cycle_index"]) for row in rows]
    if cycles != list(range(len(rows))):
        failures.append("cycle_index가 0부터 연속적이지 않음")

    kinds = Counter(row["acquisition_kind"] for row in rows)
    command_rate = int(metadata["command_rate_hz"])
    state_slots = int(metadata["state_read_rate_hz"])
    derived_error = sum(
        cycle % command_rate >= state_slots for cycle in range(expected)
    )
    expected_error = int(
        metadata.get("expected_hardware_error_samples", derived_error)
    )
    expected_state = int(
        metadata.get("expected_state_samples", expected - expected_error)
    )
    if kinds["state"] != expected_state or kinds["hardware_error"] != expected_error:
        failures.append(
            "수신 슬롯 actual="
            f"{dict(kinds)}, expected=state {expected_state}/error {expected_error}"
        )

    hardware_errors = {
        joint: [
            int(row[f"{joint}/hardware_error"])
            for row in rows
            if row[f"{joint}/hardware_error"] != ""
            and int(row[f"{joint}/hardware_error"]) != 0
        ]
        for joint in MUJOCO_DOF_ORDER
    }
    hardware_errors = {joint: values for joint, values in hardware_errors.items() if values}
    if hardware_errors:
        failures.append(f"Hardware Error={hardware_errors}")

    overrun_ms = [int(row["overrun_ns"]) / 1e6 for row in rows]
    overrun_count = sum(value > 0.0 for value in overrun_ms)
    if overrun_count:
        failures.append(f"deadline overrun {overrun_count}/{len(rows)}")

    print(f"Run: {run_dir}")
    print(f"Metadata valid_flag: {metadata.get('valid_flag')}")
    print(f"Rows: {len(rows)}/{expected}")
    print(f"Acquisition: {dict(kinds)}")
    print(f"Phases: {dict(Counter(row['phase'] for row in rows))}")
    if len(rows) >= 2:
        tx = [int(row["tx_time_ns"]) for row in rows]
        interval_ms = [(b - a) / 1e6 for a, b in zip(tx, tx[1:])]
        print(f"TX interval ms: {_fmt(_summary(interval_ms))}")
        print(f"Effective rate: {1000.0 / statistics.mean(interval_ms):.6f} Hz")
    for kind in ("state", "hardware_error"):
        selected = [row for row in rows if row["acquisition_kind"] == kind]
        if selected:
            io_ms = [
                (int(row["rx_time_ns"]) - int(row["tx_time_ns"])) / 1e6
                for row in selected
            ]
            print(f"{kind} I/O ms: {_fmt(_summary(io_ms))}")
    print(f"Deadline overrun: {overrun_count}/{len(rows)}, max={max(overrun_ms):.4f} ms")
    print(f"Hardware Error: {hardware_errors or 'none'}")

    state_rows = [row for row in rows if row["acquisition_kind"] == "state"]
    for field, label in (
        ("input_voltage_V", "Voltage V"),
        ("temperature_C", "Temperature C"),
        ("current_A", "Current A"),
        ("pwm_percent", "PWM %"),
    ):
        values = [
            float(row[f"{joint}/{field}"])
            for row in state_rows
            for joint in MUJOCO_DOF_ORDER
        ]
        print(f"{label}: {_fmt(_summary(values))}, maxabs={max(map(abs, values)):.4f}")

    if failures:
        print("RESULT: INVALID")
        for failure in failures:
            print(f"  - {failure}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("RESULT: STRUCTURE/TIMING PASS")


if __name__ == "__main__":
    main()
