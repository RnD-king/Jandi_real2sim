"""Immutable logical-run attempt allocation and explicit valid selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttemptStatus:
    logical_directory: Path
    attempts: tuple[Path, ...]
    valid_attempts: tuple[Path, ...]
    invalid_attempts: tuple[Path, ...]
    incomplete_attempts: tuple[Path, ...]
    selected_attempt: Path | None

    @property
    def state(self) -> str:
        if len(self.valid_attempts) > 1 and self.selected_attempt is None:
            return "MULTIPLE_VALID_ATTEMPTS"
        if self.selected_attempt is not None:
            return "VALID"
        if self.invalid_attempts or self.incomplete_attempts:
            return "INVALID"
        return "NOT_RUN"


def _metadata(path: Path) -> dict | None:
    file = path / "metadata.json"
    if not file.is_file():
        return None
    try:
        value = json.loads(file.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def attempt_directories(logical: Path) -> tuple[Path, ...]:
    if not logical.is_dir():
        return ()
    return tuple(sorted((path for path in logical.glob("attempt_[0-9][0-9][0-9]") if path.is_dir()),
                        key=lambda path: path.name))


def inspect_logical_run(logical: Path) -> AttemptStatus:
    attempts = attempt_directories(logical)
    # Read-only compatibility for canonical data collected before attempt directories.
    if (logical / "metadata.json").is_file():
        attempts = (logical, *attempts)
    valid, invalid, incomplete = [], [], []
    for attempt in attempts:
        metadata = _metadata(attempt)
        if metadata is None:
            incomplete.append(attempt)
        elif metadata.get("valid_flag") is True:
            valid.append(attempt)
        else:
            invalid.append(attempt)
    selected = None
    pointer = logical / "selected_attempt.txt"
    if pointer.is_file():
        name = pointer.read_text().strip()
        candidate = logical if name == "." else logical / name
        if candidate not in valid:
            raise ValueError(f"selected attempt가 valid attempt가 아닙니다: {pointer}: {name}")
        selected = candidate
    elif len(valid) == 1:
        selected = valid[0]
    return AttemptStatus(logical, attempts, tuple(valid), tuple(invalid), tuple(incomplete), selected)


def allocate_attempt(logical: Path) -> tuple[Path, int, str | None]:
    """Atomically reserve the next immutable attempt directory."""
    logical.mkdir(parents=True, exist_ok=True)
    existing = attempt_directories(logical)
    index = max((int(path.name.removeprefix("attempt_")) for path in existing), default=0) + 1
    while True:
        target = logical / f"attempt_{index:03d}"
        try:
            target.mkdir(exist_ok=False)
            break
        except FileExistsError:
            index += 1
    retry_of = existing[-1].name if existing else ("." if (logical / "metadata.json").is_file() else None)
    return target, index, retry_of


def select_valid_attempt(logical: Path) -> Path:
    status = inspect_logical_run(logical)
    if status.selected_attempt is not None:
        return status.selected_attempt
    if len(status.valid_attempts) > 1:
        names = ", ".join(path.name for path in status.valid_attempts)
        raise ValueError(
            f"multiple valid attempts: {logical}: {names}. "
            "selected_attempt.txt에 사용할 attempt 이름을 명시하십시오."
        )
    raise FileNotFoundError(f"selected valid attempt가 없습니다: {logical}")


def write_attempt_selection(logical: Path, attempt_name: str) -> Path:
    status = inspect_logical_run(logical)
    candidate = logical / attempt_name
    if candidate not in status.valid_attempts:
        raise ValueError(f"선택 대상은 valid attempt여야 합니다: {candidate}")
    pointer = logical / "selected_attempt.txt"
    temporary = logical / ".selected_attempt.tmp"
    temporary.write_text(attempt_name + "\n")
    temporary.replace(pointer)
    return pointer
