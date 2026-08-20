"""Result records. No result exists unless it is on disk.

Every experiment writes one JSON file holding the full configuration, the git
commit the code was at, the seed, the wall clock cost, and the environment the
numbers came from.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_RESULTS_DIR = Path("results")


def git_commit(repo: str | Path = ".") -> str:
    """Return the current git commit, with a dirty marker when the tree has changes."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return f"{commit}-dirty" if status else commit


def environment() -> dict[str, Any]:
    """Return the library versions and hardware the numbers were produced on."""
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }


@dataclass
class ResultRecord:
    """One experiment's record, written to disk as JSON."""

    name: str
    config: dict[str, Any]
    seed: int
    payload: dict[str, Any] = field(default_factory=dict)
    git_commit: str = field(default_factory=git_commit)
    environment: dict[str, Any] = field(default_factory=environment)
    started_at: str = ""
    wall_clock_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        return {
            "name": self.name,
            "git_commit": self.git_commit,
            "seed": self.seed,
            "started_at": self.started_at,
            "wall_clock_seconds": self.wall_clock_seconds,
            "environment": self.environment,
            "config": self.config,
            "payload": self.payload,
        }

    def write(self, results_dir: str | Path = DEFAULT_RESULTS_DIR) -> Path:
        """Write the record to a timestamped JSON file and return its path."""
        directory = Path(results_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = (self.started_at or _now()).replace(":", "").replace("-", "")
        path = directory / f"{self.name}__{stamp}__{self.git_commit[:8]}.json"
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=False, default=str))
        return path


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@contextmanager
def record(
    name: str,
    config: dict[str, Any],
    seed: int,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> Iterator[ResultRecord]:
    """Open a result record, time the work inside, and write it on the way out.

    The record is written even when the work raises, with the error in the
    payload, so a failed run leaves evidence rather than nothing.
    """
    entry = ResultRecord(name=name, config=config, seed=seed, started_at=_now())
    start = time.perf_counter()
    try:
        yield entry
    except BaseException as error:  # noqa: BLE001 - the record must survive any failure
        entry.payload["error"] = f"{type(error).__name__}: {error}"
        entry.wall_clock_seconds = time.perf_counter() - start
        entry.write(results_dir)
        raise
    entry.wall_clock_seconds = time.perf_counter() - start
    entry.write(results_dir)


def load_records(results_dir: str | Path = DEFAULT_RESULTS_DIR, name: str | None = None) -> list[dict[str, Any]]:
    """Return every result record on disk, optionally filtered by experiment name."""
    directory = Path(results_dir)
    if not directory.exists():
        return []
    loaded: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        entry = json.loads(path.read_text())
        if name is None or entry.get("name") == name:
            entry["_path"] = str(path)
            loaded.append(entry)
    return loaded
