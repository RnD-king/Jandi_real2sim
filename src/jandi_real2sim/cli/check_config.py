from __future__ import annotations

import argparse

from .common import add_config_argument, load_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Jandi Real2Sim config 및 100 Hz 계약 확인")
    add_config_argument(parser)
    args = parser.parse_args()
    config = load_from_args(args)
    print(f"Config: {config.source}")
    print(
        f"Bus: {config.bus.port}, {config.bus.baudrate} baud, "
        f"command={config.bus.command_rate_hz} Hz, "
        f"state/error={config.bus.state_read_rate_hz}/"
        f"{config.bus.hardware_error_read_rate_hz} slots/s"
    )
    print("Timing: physics=500 Hz 예정, command=100 Hz, state/error=99/1 Hz, policy=50 Hz 별도")
    print("Walking pose (MuJoCo order):")
    for joint in config.joints:
        status = "READY" if joint.hardware_ready else "BLOCKED"
        print(
            f"  {joint.name:9s} {joint.walking_rad:+.3f} rad  "
            f"range=[{joint.min_rad:+.2f},{joint.max_rad:+.2f}]  "
            f"id={joint.motor_id} zero={joint.zero_tick} dir={joint.direction} {status}"
        )
    print(f"Hardware execution: {'READY' if config.hardware_ready else 'BLOCKED'}")


if __name__ == "__main__":
    main()
