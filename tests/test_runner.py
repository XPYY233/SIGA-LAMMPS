"""Tests for Level 2 — running the references in real LAMMPS.

The centrepiece is that every task's reference solution must pass **both** level 1
(structural) and level 2 (runtime). A task whose own reference cannot pass is
unsatisfiable, and would fail every agent for a reason no adapter could fix.

Skipped cleanly when no LAMMPS executable is available.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from adapter.runner import logs_show_success, run_local
from adapter.tasks import load_task
from adapter.validator import validate_workspace

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "benchmark" / "tasks"
GROUND_TRUTH = REPO_ROOT / "benchmark" / "ground_truth"

TASK_IDS = (
    "lj_melt",
    "nvt_equilibration",
    "msd_diffusion",
    "uniaxial_tension",
    "nanoindentation",
)


def _lammps_available() -> bool:
    from config.loader import load_settings

    try:
        binary = load_settings().lammps.local_bin
    except Exception:  # noqa: BLE001 - a broken config means "cannot test here"
        return False
    return shutil.which(binary) is not None


requires_lammps = pytest.mark.skipif(not _lammps_available(), reason="no LAMMPS executable available")


# --------------------------------------------------------------------------- #
# the success predicate
# --------------------------------------------------------------------------- #


def test_a_clean_log_is_success() -> None:
    text = "Created 100 atoms\nLoop time of 0.1\nTotal wall time: 0:00:01\n"
    assert logs_show_success(text, 0)[0] is True


def test_nonzero_exit_is_not_success() -> None:
    ok, error = logs_show_success("ERROR: Lost atoms: original 10 current 8\n", 1)
    assert ok is False
    assert error is not None and "Lost atoms" in error


def test_a_comment_quoting_an_error_is_not_a_failure() -> None:
    """The regression that matters.

    LAMMPS echoes the input script into its log, so a *comment* mentioning a
    failure mode appears in the log verbatim. Searching for the phrase reported a
    failure that never happened, on a script that ran perfectly — which is why
    success is judged on the ERROR prefix and the exit code instead.
    """
    text = (
        "# LAMMPS reports \"Lost atoms: original 648 current 624\" when the bottom is unanchored\n"
        "Created 648 atoms\n"
        "Total wall time: 0:00:01\n"
    )
    assert logs_show_success(text, 0)[0] is True


def test_a_missing_completion_marker_is_not_success() -> None:
    ok, error = logs_show_success("Created 10 atoms\n", 0)
    assert ok is False
    assert error is not None and "did not finish" in error


# --------------------------------------------------------------------------- #
# the references
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_reference_passes_its_own_structural_checks(tmp_path: Path, task_id: str) -> None:
    """Level 1. A reference that fails its own task means the task is wrong."""
    task = load_task(task_id, TASKS_DIR)
    assert task.reference is not None
    source = GROUND_TRUTH / task.reference
    assert source.is_file(), f"reference missing: {source}"
    workspace = tmp_path / task_id
    workspace.mkdir()
    shutil.copy(source, workspace / "in.test")
    result = validate_workspace(workspace, task=task)
    assert result.valid, f"{task_id} reference fails its own task: {result.render()}"


@requires_lammps
@pytest.mark.real_corpus
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_reference_actually_runs_in_lammps(tmp_path: Path, task_id: str) -> None:
    """Level 2. Structural validity does not imply the simulator accepts it.

    This is not a formality. Two drafts of the nanoindentation reference passed
    every structural check while LAMMPS refused to run them, for two unrelated
    reasons — an indenter overlapping the substrate at t=0, and an unanchored
    slab bleeding atoms out of an open boundary. Neither was visible without
    executing.
    """
    task = load_task(task_id, TASKS_DIR)
    assert task.reference is not None
    workspace = tmp_path / task_id
    workspace.mkdir()
    shutil.copy(GROUND_TRUTH / task.reference, workspace / "in.test")

    outcome = run_local(workspace, timeout=900)
    assert outcome.ran, f"{task_id} failed to run: {outcome.error}"
    assert outcome.completed
    assert outcome.facts.get("atoms", 0) > 0, "the run created no atoms"


@requires_lammps
def test_run_local_reports_a_failure_as_an_outcome_not_an_exception(tmp_path: Path) -> None:
    """The evaluator must be able to record *how* a run failed."""
    (tmp_path / "in.test").write_text(
        "units lj\natom_style atomic\nlattice fcc 0.8442\n"
        "region box block 0 2 0 2 0 2\ncreate_box 1 box\ncreate_atoms 1 box\n"
        "mass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0 2.5\n"
        "fix 1 all nve\ntimestep 500.0\nrun 10\n",
        encoding="utf-8",
    )
    outcome = run_local(tmp_path, timeout=120)
    assert outcome.ran is False
    assert outcome.error is not None
    assert outcome.exit_code != 0


@requires_lammps
def test_run_local_raises_when_there_is_nothing_to_run(tmp_path: Path) -> None:
    """No script is a different condition from a failing script."""
    with pytest.raises(FileNotFoundError):
        run_local(tmp_path)


def test_missing_lammps_binary_is_reported_clearly(tmp_path: Path) -> None:
    (tmp_path / "in.test").write_text("units lj\nrun 0\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="not found"):
        run_local(tmp_path, lammps_bin="definitely-not-a-lammps-binary")


@requires_lammps
def test_run_local_records_timing_and_facts(tmp_path: Path) -> None:
    shutil.copy(GROUND_TRUTH / "lj_melt" / "in.melt", tmp_path / "in.test")
    outcome = run_local(tmp_path, timeout=300)
    assert outcome.ran
    assert outcome.seconds > 0
    assert outcome.facts.get("atoms") == 256
    assert outcome.log_path is not None and outcome.log_path.is_file()
