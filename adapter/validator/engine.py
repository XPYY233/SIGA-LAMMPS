"""Validation orchestration — the single entry point for X.

Every consumer reaches validation through here: the agent's
``validate_lammps_input`` tool, the harness stop gate ``S``, and the benchmark's
level-1 scoring. That is deliberate. The paper describes its LAMMPS validator as
applying *the same checks as the hook*, and the only way to make that identity
structural rather than a promise is for all three to call one function.

If the agent were gated on one rule set and scored on another, the ablation would
be uninterpretable: a run could be blocked for a reason the score never sees.
"""

from __future__ import annotations

import re
from pathlib import Path

from adapter.tasks import TaskSpec, load_task
from adapter.validator.findings import ValidationResult
from adapter.validator.rules import RuleContext, run_rules
from adapter.validator.script import parse_script, summarise

__all__ = [
    "INVALID_EXIT",
    "VALID_EXIT",
    "find_input_script",
    "validate_script",
    "validate_workspace",
]

#: Exit codes shared with `adapter/cli.py`. They distinguish "the input is
#: invalid" from "the validator could not run", because the correct response
#: differs: the first blocks the agent, the second alerts a human.
VALID_EXIT = 0
INVALID_EXIT = 1

#: Filenames LAMMPS input scripts conventionally use.
_SCRIPT_PATTERNS = ("in.*", "*.lmp", "in", "input.*")


def find_input_script(workspace: Path | str) -> Path | None:
    """Locate the input script in a workspace.

    Returns ``None`` when the workspace holds no candidate, or holds several
    genuinely ambiguous ones. Guessing between two scripts would validate the
    wrong file and report a confident result about it, so ambiguity is reported
    rather than resolved.
    """
    workspace = Path(workspace)
    if not workspace.is_dir():
        return None

    for pattern in _SCRIPT_PATTERNS:
        candidates = sorted(p for p in workspace.glob(pattern) if p.is_file())
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            # `in.melt` and `in.melt.gpu` are a real LAMMPS pattern; prefer the
            # shortest name, which is the canonical one in every example set.
            return sorted(candidates, key=lambda p: (len(p.name), p.name))[0]
    return None


def validate_script(
    text: str,
    *,
    workspace: Path | str | None = None,
    task: TaskSpec | None = None,
    supported_atom_styles: tuple[str, ...] = (),
    supported_unit_styles: tuple[str, ...] = (),
    max_lines: int = 400,
) -> ValidationResult:
    """Validate one script's text.

    Args:
        text: the LAMMPS input script.
        workspace: directory used to resolve file references. ``None`` disables
            existence checks, which become "unverified" suggestions instead of
            errors — an absent workspace must not manufacture failures.
        task: the frozen task specification, when validating against a benchmark.
    """
    workspace_path = Path(workspace) if workspace is not None else None
    script = parse_script(text)
    context = RuleContext(
        script=script,
        workspace=workspace_path,
        task=task,
        supported_atom_styles=supported_atom_styles,
        supported_unit_styles=supported_unit_styles,
        max_lines=max_lines,
    )
    run_rules(context)

    facts = summarise(script)
    facts.update(context.facts)
    facts["workspace"] = str(workspace_path) if workspace_path else None
    facts["task"] = task.id if task is not None else None
    if workspace_path is not None:
        found = find_input_script(workspace_path)
        facts["input_script"] = found.name if found else None

    return ValidationResult(
        findings=tuple(context.findings),
        task_id=task.id if task is not None else None,
        script=text,
        facts=facts,
    )


def validate_workspace(
    workspace: Path | str,
    *,
    task: TaskSpec | str | None = None,
    tasks_dir: Path | str | None = None,
    supported_atom_styles: tuple[str, ...] = (),
    supported_unit_styles: tuple[str, ...] = (),
    max_lines: int = 400,
) -> ValidationResult:
    """Validate the input script found in *workspace*.

    Raises:
        FileNotFoundError: the workspace is missing, or holds no input script.
            Neither is a validation *finding*: they mean there is nothing to
            validate, which the caller must distinguish from "invalid".
    """
    workspace = Path(workspace)
    if not workspace.is_dir():
        raise FileNotFoundError(f"workspace not found: {workspace}")

    script_path = find_input_script(workspace)
    if script_path is None:
        raise FileNotFoundError(
            f"no LAMMPS input script in {workspace} (looked for {', '.join(_SCRIPT_PATTERNS)})"
        )

    spec: TaskSpec | None
    if isinstance(task, TaskSpec):
        spec = task
    elif isinstance(task, str):
        if tasks_dir is None:
            raise ValueError("tasks_dir is required when naming a task by id")
        spec = load_task(task, tasks_dir)
    else:
        spec = None

    return validate_script(
        script_path.read_text(encoding="utf-8", errors="replace"),
        workspace=workspace,
        task=spec,
        supported_atom_styles=supported_atom_styles,
        supported_unit_styles=supported_unit_styles,
        max_lines=max_lines,
    )


def count_unverified(result: ValidationResult) -> int:
    """Findings that report an *inability* to check, rather than a defect.

    Reported separately so a run is never described as fully verified when part
    of it could not be examined — the same honesty the design principles require
    of level 4.
    """
    pattern = re.compile(r"_UNVERIFIED$|_UNVERIFIED_")
    return sum(1 for f in result.findings if pattern.search(f.code))
