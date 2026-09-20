#!/usr/bin/env python3
"""Validate the fixed baseline and isolated empty research workspaces."""

import hashlib
from pathlib import Path

from experiment import BATCH_SIZE, EPOCHS, LR, MLP

ROOT = Path(__file__).resolve().parent


def main() -> None:
    assert sum(parameter.numel() for parameter in MLP().parameters()) == 269322
    assert (BATCH_SIZE, EPOCHS, LR) == (128, 5, 1e-3)
    assert (ROOT / "baseline" / "attempt.py").is_file(), "Missing fixed baseline"
    assert not (ROOT / "RESEARCH_BASELINE.md").exists()

    infrastructure_hash = hashlib.sha256((ROOT / "experiment.py").read_bytes()).hexdigest()
    sessions = (
        ("direct_session", "RESEARCH_DIRECT.md"),
        ("analogy_session", "RESEARCH_ANALOGY.md"),
    )
    for directory_name, prompt_name in sessions:
        directory = ROOT / directory_name
        assert directory.is_dir()
        assert {path.name for path in directory.iterdir()} == {"experiment.py", prompt_name, "attempts"}
        shared_experiment = directory / "experiment.py"
        assert shared_experiment.is_symlink(), f"{directory_name} must use the shared experiment.py"
        assert shared_experiment.resolve() == (ROOT / "experiment.py").resolve()
        assert hashlib.sha256(shared_experiment.read_bytes()).hexdigest() == infrastructure_hash
        attempts = directory / "attempts"
        assert attempts.is_dir() and not any(attempts.iterdir()), f"Prewritten attempts in {directory_name}"

    root_attempts = ROOT / "attempts"
    assert not root_attempts.exists() or not any(root_attempts.iterdir())
    print("Validated fixed baseline and isolated empty direct/analogy workspaces.")


if __name__ == "__main__":
    main()
