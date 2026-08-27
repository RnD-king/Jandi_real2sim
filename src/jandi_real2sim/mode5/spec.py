"""Canonical Mode-5 experiment constants from the project README.

This module deliberately contains interface constants only.  Physical values
that have not been measured belong in YAML and must remain ``null``.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAMPAIGN = PROJECT_ROOT / "configs" / "mode5" / "campaign.yaml"

ARM_LENGTHS = ("L1", "L2")
LOADS = ("m250", "m500", "m750")
MECHANICAL_CONFIGURATIONS = tuple(
    f"{length}_{load}" for length in ARM_LENGTHS for load in LOADS
)
MAIN_TRAJECTORIES = (
    "accelerated_oscillation",
    "slow_plus_highfreq",
    "slowly_raise_lower",
)
APPROACH_DIRECTIONS = ("approach_positive", "approach_negative")
REPEATS = (1, 2, 3)

# One source of truth for every hardware-execution confirmation string.
CONFIRMATIONS = {
    "pilot": "PILOT_MX106_MODE5",
    "static": "CALIBRATE_MX106_MODE5",
    "delay": "CALIBRATE_DELAY_MX106_MODE5",
    "collect": "COLLECT_MX106_MODE5",
}

STATIC_RUN_COUNT = (
    len(MECHANICAL_CONFIGURATIONS) * len(APPROACH_DIRECTIONS) * len(REPEATS)
)
DYNAMIC_RUN_COUNT = (
    len(MECHANICAL_CONFIGURATIONS) * len(MAIN_TRAJECTORIES) * len(REPEATS)
)

# ROBOTIS MX-106R(2.0): Realtime Tick(120) is 1 ms/count and wraps after
# 32767.  This is deliberately not a generic uint16 modulus.
REALTIME_TICK_MODULUS = 32768

# The equivalent MuJoCo pendulum is defined in one canonical coordinate.
CANONICAL_HINGE_AXIS = (0.0, 1.0, 0.0)
