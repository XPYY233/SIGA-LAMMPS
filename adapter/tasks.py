"""Frozen benchmark task specifications.

Each task is one YAML file under ``benchmark/tasks/`` carrying the researcher's
natural-language request plus the deterministic expectations the four evaluation
levels need.

**Task requirements live here, not in the validator.** The alternative — having
X infer required commands from the natural-language specification — was rejected:
it would be non-deterministic, unauditable, and would put a second model in the
validation path, which the design principles rule out. The specification text is
what the *agent* reads; this file's structured blocks are what the *evaluator*
and the validator read. They are two views of one frozen artefact.

**Specifications are frozen.** Every adapter configuration is scored against
these exact strings. If vanilla and M+R+X+S saw different task text, the ablation
would compare tasks rather than adapters.

Check kinds are validated on load. A misspelled kind would otherwise be silently
skipped, and a check that never runs is worse than one that fails: it reports
success for a run that never satisfied it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "CHECK_KINDS_LEVEL3",
    "CHECK_KINDS_LEVEL4",
    "TaskSpec",
    "TaskSpecError",
    "list_tasks",
    "load_task",
    "load_tasks",
]

#: Check kinds level 3 (task compliance) may use.
CHECK_KINDS_LEVEL3 = frozenset(
    {
        "regex",  # a pattern must appear in the script
        "regex_any",  # at least one of several patterns must appear
        "pattern_number",  # `command <arg_index>` must equal an expected number
        "regex_value",  # a named token inside a command line must equal a value
        "damping_ratio",  # a damping constant must scale with the timestep
        "run_steps_min",  # the run must be at least this long
        "lattice_density",  # lattice must express the requested reduced density
        "fix_group_distinct",  # a fix must not target the whole system
    }
)

#: Check kinds level 4 (physical sanity) may use.
CHECK_KINDS_LEVEL4 = frozenset(
    {
        "log_range",  # a thermo column must stay in a band after equilibration
        "log_forbidden",  # a failure pattern must NOT appear in the LAMMPS log
        "log_finite",  # named columns must contain no NaN or infinity
        "log_series_increasing",  # a series must actually grow (e.g. MSD)
        "log_stability",  # a column must not drift beyond a relative bound
        "human_review_required",  # explicitly not judged automatically
    }
)


class TaskSpecError(ValueError):
    """A task specification is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class Level1Requirements:
    """Structural requirements, evaluated by the shared validator."""

    required_commands: tuple[str, ...]
    required_patterns: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TaskSpec:
    """One frozen benchmark task."""

    id: str
    title: str
    summary: str
    specification: str
    reference: str | None
    required_files: tuple[str, ...]
    level1: Level1Requirements
    level3: tuple[dict[str, Any], ...]
    level4: tuple[dict[str, Any], ...]
    notes: str = ""
    path: Path | None = field(default=None, compare=False)

    @property
    def review_required(self) -> tuple[dict[str, Any], ...]:
        """Checks explicitly marked as needing a human.

        Surfaced separately so a report can never present a run as fully
        verified while some physical question was simply not judged.
        """
        return tuple(c for c in self.level4 if c.get("kind") == "human_review_required")


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskSpecError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _check_list(value: Any, where: str, allowed: frozenset[str]) -> tuple[dict[str, Any], ...]:
    """Validate one level's checks, rejecting unknown kinds and duplicate ids."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TaskSpecError(f"{where}: expected a list of checks")
    checks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        check = _require_mapping(raw, f"{where}[{index}]")
        kind = check.get("kind")
        if kind not in allowed:
            raise TaskSpecError(
                f"{where}[{index}]: unknown kind {kind!r}. "
                f"Known kinds: {', '.join(sorted(allowed))}"
            )
        identifier = check.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise TaskSpecError(f"{where}[{index}]: every check needs a non-empty id")
        if identifier in seen:
            raise TaskSpecError(f"{where}: duplicate check id {identifier!r}")
        seen.add(identifier)
        checks.append(check)
    return tuple(checks)


def parse_task(payload: dict[str, Any], *, path: Path | None = None) -> TaskSpec:
    """Validate and convert one parsed YAML mapping into a :class:`TaskSpec`."""
    where = str(path) if path is not None else "<task>"
    payload = _require_mapping(payload, where)

    task_id = payload.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise TaskSpecError(f"{where}: 'id' is required")

    specification = payload.get("specification")
    if not isinstance(specification, str) or not specification.strip():
        # Without it there is nothing to give the agent, and the run is not a
        # benchmark task at all.
        raise TaskSpecError(f"{where}: 'specification' (the natural-language request) is required")

    level1_raw = _require_mapping(payload.get("level1") or {}, f"{where}: level1")
    commands = level1_raw.get("required_commands") or []
    if not isinstance(commands, list) or not all(isinstance(c, str) for c in commands):
        raise TaskSpecError(f"{where}: level1.required_commands must be a list of strings")

    patterns = level1_raw.get("required_patterns") or []
    if not isinstance(patterns, list):
        raise TaskSpecError(f"{where}: level1.required_patterns must be a list")
    for index, raw in enumerate(patterns):
        entry = _require_mapping(raw, f"{where}: level1.required_patterns[{index}]")
        if not entry.get("id") or not entry.get("pattern"):
            raise TaskSpecError(
                f"{where}: level1.required_patterns[{index}] needs both 'id' and 'pattern'"
            )

    level3 = _check_list(payload.get("level3"), f"{where}: level3", CHECK_KINDS_LEVEL3)
    level4 = _check_list(payload.get("level4"), f"{where}: level4", CHECK_KINDS_LEVEL4)
    if not level3:
        raise TaskSpecError(f"{where}: level3 needs at least one task-compliance check")
    if not level4:
        raise TaskSpecError(f"{where}: level4 needs at least one physical check")

    files = payload.get("required_files") or []
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        raise TaskSpecError(f"{where}: required_files must be a list of strings")

    reference = payload.get("reference")

    return TaskSpec(
        id=task_id,
        title=str(payload.get("title") or task_id),
        summary=str(payload.get("summary") or ""),
        specification=specification.strip(),
        reference=str(reference) if reference else None,
        required_files=tuple(files),
        level1=Level1Requirements(
            required_commands=tuple(commands),
            required_patterns=tuple(dict(p) for p in patterns),
        ),
        level3=level3,
        level4=level4,
        notes=str(payload.get("notes") or "").strip(),
        path=path,
    )


def load_task(task_id: str, tasks_dir: Path | str) -> TaskSpec:
    """Load one task by id.

    Raises:
        TaskSpecError: the file is absent (with the available ids listed), or
            fails validation.
    """
    tasks_dir = Path(tasks_dir)
    path = tasks_dir / f"{task_id}.yaml"
    if not path.is_file():
        available = ", ".join(sorted(t.id for t in load_tasks(tasks_dir))) or "none"
        raise TaskSpecError(f"no task {task_id!r} in {tasks_dir}. Available: {available}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TaskSpecError(f"{path}: invalid YAML: {exc}") from exc
    spec = parse_task(payload, path=path)
    if spec.id != task_id:
        raise TaskSpecError(
            f"{path}: file name says {task_id!r} but the spec declares id {spec.id!r}"
        )
    return spec


def load_tasks(tasks_dir: Path | str) -> list[TaskSpec]:
    """Load every task in a directory, in stable id order.

    Raises:
        TaskSpecError: any file in the directory fails validation. A broken task
            is reported rather than skipped, because silently dropping one would
            shrink the benchmark and change every aggregate.
    """
    tasks_dir = Path(tasks_dir)
    if not tasks_dir.is_dir():
        raise TaskSpecError(f"task directory not found: {tasks_dir}")
    specs: list[TaskSpec] = []
    for path in sorted(tasks_dir.glob("*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise TaskSpecError(f"{path}: invalid YAML: {exc}") from exc
        spec = parse_task(payload, path=path)
        if spec.id != path.stem:
            raise TaskSpecError(
                f"{path}: file name says {path.stem!r} but the spec declares id {spec.id!r}"
            )
        specs.append(spec)
    if not specs:
        raise TaskSpecError(f"no task specifications found in {tasks_dir}")
    return sorted(specs, key=lambda s: s.id)


def list_tasks(tasks_dir: Path | str) -> list[TaskSpec]:
    """Alias for :func:`load_tasks`, kept for readability at call sites."""
    return load_tasks(tasks_dir)
