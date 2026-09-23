"""Tests for the frozen benchmark task specifications.

The properties that matter are not the YAML plumbing but the guarantees the
ablation rests on: every task loads, ids match filenames, the specifications are
usable as agent-facing text, and a malformed check is rejected rather than
silently skipped.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from adapter.tasks import (
    CHECK_KINDS_LEVEL3,
    CHECK_KINDS_LEVEL4,
    TaskSpecError,
    load_task,
    load_tasks,
    parse_task,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "benchmark" / "tasks"

#: The five classes in scope for v1, from the project brief.
EXPECTED_TASK_IDS = (
    "lj_melt",
    "msd_diffusion",
    "nanoindentation",
    "nvt_equilibration",
    "uniaxial_tension",
)


@pytest.fixture(scope="module")
def tasks() -> list:
    return load_tasks(TASKS_DIR)


# --------------------------------------------------------------------------- #
# the real task set
# --------------------------------------------------------------------------- #


def test_every_expected_task_exists(tasks: list) -> None:
    assert tuple(t.id for t in tasks) == EXPECTED_TASK_IDS


def test_task_id_matches_its_filename(tasks: list) -> None:
    for task in tasks:
        assert task.path is not None
        assert task.path.stem == task.id


def test_specifications_are_usable_agent_facing_text(tasks: list) -> None:
    """The specification is what the agent is given, so it must stand alone."""
    for task in tasks:
        text = task.specification
        assert len(text) > 200, f"{task.id}: specification is too thin to be a task"
        assert not text.startswith("#"), f"{task.id}: specification looks like markup, not a request"
        # It must read as a request, not as a restatement of the checker.
        assert any(
            word in text.lower() for word in ("simulate", "equilibrate", "measure", "perform")
        ), f"{task.id}: specification does not read as an instruction"


def test_specifications_are_frozen_and_distinct(tasks: list) -> None:
    """Distinct text per task; identical specs would collapse the benchmark."""
    texts = [t.specification for t in tasks]
    assert len(set(texts)) == len(texts), "two tasks share an identical specification"


def test_every_task_carries_all_four_levels(tasks: list) -> None:
    for task in tasks:
        assert task.level1.required_commands, f"{task.id}: no level-1 required commands"
        assert task.level1.required_patterns, f"{task.id}: no level-1 required patterns"
        assert task.level3, f"{task.id}: no level-3 compliance checks"
        assert task.level4, f"{task.id}: no level-4 physical checks"


def test_structural_commands_are_lowercase_keywords(tasks: list) -> None:
    """LAMMPS keywords are lowercase; a capitalised entry would never match."""
    for task in tasks:
        for command in task.level1.required_commands:
            assert command == command.lower(), f"{task.id}: {command!r} must be lowercase"


def test_level1_patterns_compile_and_name_themselves(tasks: list) -> None:
    import re

    for task in tasks:
        for entry in task.level1.required_patterns:
            assert entry.get("description"), f"{task.id}/{entry['id']}: no description"
            try:
                re.compile(entry["pattern"])
            except re.error as exc:  # pragma: no cover - only on a bad edit
                pytest.fail(f"{task.id}/{entry['id']}: invalid regex: {exc}")


def test_every_check_has_an_id_and_description_where_required(tasks: list) -> None:
    for task in tasks:
        for check in (*task.level3, *task.level4):
            assert check.get("id"), f"{task.id}: a check has no id"
            # `human_review_required` is the one kind that needs no criterion —
            # its whole point is that no automatic criterion exists.
            if check["kind"] != "human_review_required":
                assert check.get("description") or check.get("expected") is not None, (
                    f"{task.id}/{check['id']}: a check needs a description or an expectation"
                )


def test_physically_unknowable_questions_are_flagged_for_review(tasks: list) -> None:
    """Each task must admit what it cannot judge automatically.

    The design principles forbid an LLM asserting physical correctness without a
    reliable basis. A task with no `human_review_required` marker is claiming it
    checks everything, which these tasks cannot.
    """
    for task in tasks:
        assert task.review_required, f"{task.id}: claims full automatic verification"
        for check in task.review_required:
            assert check.get("description"), f"{task.id}/{check['id']}: review item needs a reason"


def test_reference_paths_are_relative_and_named(tasks: list) -> None:
    for task in tasks:
        if task.reference is None:
            continue
        assert not task.reference.startswith("/"), f"{task.id}: reference must be relative"
        assert "/" in task.reference, f"{task.id}: reference should be <task>/<file>"


# --------------------------------------------------------------------------- #
# loader strictness — a check that silently skips is worse than one that fails
# --------------------------------------------------------------------------- #


def _minimal(**overrides: object) -> dict:
    base = {
        "id": "probe",
        "specification": "Simulate something reasonable and report it.",
        "level1": {"required_commands": ["units"], "required_patterns": [{"id": "a", "pattern": "^x"}]},
        "level3": [{"id": "c", "kind": "regex", "pattern": "^y", "description": "d"}],
        "level4": [{"id": "p", "kind": "log_finite", "columns": ["TotEng"]}],
    }
    base.update(overrides)
    return base


def test_minimal_spec_parses() -> None:
    spec = parse_task(_minimal())
    assert spec.id == "probe"
    assert spec.level1.required_commands == ("units",)


def test_unknown_check_kind_is_rejected() -> None:
    """The whole reason kinds are validated: a typo must not skip a check."""
    with pytest.raises(TaskSpecError, match="unknown kind"):
        parse_task(_minimal(level3=[{"id": "c", "kind": "regexx", "pattern": "^y"}]))


def test_unknown_level4_kind_is_rejected() -> None:
    with pytest.raises(TaskSpecError, match="unknown kind"):
        parse_task(_minimal(level4=[{"id": "p", "kind": "looks_fine_to_me"}]))


def test_duplicate_check_id_is_rejected() -> None:
    with pytest.raises(TaskSpecError, match="duplicate check id"):
        parse_task(
            _minimal(
                level3=[
                    {"id": "c", "kind": "regex", "pattern": "^y"},
                    {"id": "c", "kind": "regex", "pattern": "^z"},
                ]
            )
        )


def test_missing_specification_is_rejected() -> None:
    """Without agent-facing text there is no task, only a checker."""
    with pytest.raises(TaskSpecError, match="specification"):
        parse_task(_minimal(specification="   "))


def test_empty_level3_or_level4_is_rejected() -> None:
    with pytest.raises(TaskSpecError, match="level3"):
        parse_task(_minimal(level3=[]))
    with pytest.raises(TaskSpecError, match="level4"):
        parse_task(_minimal(level4=[]))


def test_missing_id_is_rejected() -> None:
    payload = _minimal()
    del payload["id"]
    with pytest.raises(TaskSpecError, match="'id' is required"):
        parse_task(payload)


def test_load_task_reports_available_ids(tmp_path: Path) -> None:
    (tmp_path / "real.yaml").write_text(yaml.safe_dump(_minimal(id="real")), encoding="utf-8")
    with pytest.raises(TaskSpecError, match="Available: real"):
        load_task("absent", tmp_path)


def test_filename_id_mismatch_is_rejected(tmp_path: Path) -> None:
    """A mismatch means the caller and the file disagree about which task ran."""
    (tmp_path / "alpha.yaml").write_text(yaml.safe_dump(_minimal(id="beta")), encoding="utf-8")
    with pytest.raises(TaskSpecError, match="declares id"):
        load_tasks(tmp_path)


def test_a_broken_task_is_reported_not_skipped(tmp_path: Path) -> None:
    """Silently dropping a broken task would shrink the benchmark unnoticed."""
    (tmp_path / "good.yaml").write_text(yaml.safe_dump(_minimal(id="good")), encoding="utf-8")
    (tmp_path / "bad.yaml").write_text(
        textwrap.dedent(
            """
            id: bad
            specification: Simulate something.
            level3:
              - id: c
                kind: not_a_kind
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(TaskSpecError):
        load_tasks(tmp_path)


def test_missing_task_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(TaskSpecError, match="not found"):
        load_tasks(tmp_path / "absent")


def test_kind_sets_are_disjoint_and_nonempty() -> None:
    """A kind valid at both levels would make level assignment meaningless."""
    assert CHECK_KINDS_LEVEL3 and CHECK_KINDS_LEVEL4
    assert not (CHECK_KINDS_LEVEL3 & CHECK_KINDS_LEVEL4)
