"""Level 3 and level 4 evaluation.

Level 1 is the validator and level 2 is a real run; both already exist and are
reused here rather than reimplemented. This module adds the two levels that need
the task specification:

* **Level 3 — task compliance.** Did it run the task that was asked for: the
  right ensemble, the right target temperature, the right run length, the
  observable actually computed *and* output.
* **Level 4 — physical sanity**, only as far as it can be judged automatically.

Two properties are load-bearing:

* **Reference similarity is never scored.** The design principles forbid treating
  the reference deck as the only correct answer, and functionally equivalent
  LAMMPS scripts legitimately differ in fix ids, naming, ordering and numeric
  choices. Everything here checks *properties of the submission*, never its
  resemblance to a reference.
* **Level 4 admits what it cannot decide.** A physical question that needs a
  chosen analysis window is reported as `human_review_required` rather than
  guessed. An evaluator that silently marks everything it did not check as fine
  is worse than one that marks nothing at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapter.tasks import TaskSpec
from adapter.validator.script import ParsedScript, parse_script

__all__ = ["CheckOutcome", "Evaluation", "evaluate_run", "run_level3", "run_level4"]

#: LAMMPS prints its thermo table under a header row beginning with "Step".
_THERMO_HEADER = re.compile(r"^\s*Step\b")


@dataclass(frozen=True)
class CheckOutcome:
    """One check's verdict."""

    id: str
    passed: bool | None  # None means "not automatically decidable"
    detail: str = ""
    kind: str = ""
    needs_human: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "passed": self.passed,
            "detail": self.detail,
            "needs_human": self.needs_human,
        }


@dataclass
class Evaluation:
    """Everything the four levels established about one run."""

    task_id: str | None = None
    level1_valid: bool | None = None
    level1_findings: dict[str, Any] = field(default_factory=dict)
    level2_ran: bool | None = None
    level2_detail: str | None = None
    level3: list[CheckOutcome] = field(default_factory=list)
    level4: list[CheckOutcome] = field(default_factory=list)
    failure_category: str = "unknown"

    @property
    def compliance(self) -> float | None:
        """Fraction of level-3 checks that passed, or None when none applied."""
        decided = [c for c in self.level3 if c.passed is not None]
        if not decided:
            return None
        return sum(1 for c in decided if c.passed) / len(decided)

    @property
    def sanity(self) -> float | None:
        decided = [c for c in self.level4 if c.passed is not None]
        if not decided:
            return None
        return sum(1 for c in decided if c.passed) / len(decided)

    @property
    def needs_human(self) -> list[str]:
        """Checks nobody could decide automatically. Surfaced, never hidden."""
        return [c.id for c in (*self.level3, *self.level4) if c.needs_human]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task_id,
            "level1": {"valid": self.level1_valid, "findings": self.level1_findings},
            "level2": {"ran": self.level2_ran, "detail": self.level2_detail},
            "level3": [c.to_dict() for c in self.level3],
            "level4": [c.to_dict() for c in self.level4],
            "compliance": self.compliance,
            "sanity": self.sanity,
            "needs_human": self.needs_human,
            "failure_category": self.failure_category,
        }


# --------------------------------------------------------------------------- #
# helpers over a parsed script
# --------------------------------------------------------------------------- #


def _command_arguments(script: ParsedScript, keyword: str) -> list[tuple[str, ...]] | None:
    """Arguments of the first command with this keyword.

    Uses the command's *style* argument where relevant — a `fix nvt` is a fix,
    not a command called nvt — so callers ask for the fix style, not the keyword.
    """
    for command in script.commands:
        if command.keyword == keyword:
            return list(command.args)
    return None


def _fix_arguments(script: ParsedScript, style: str) -> list[tuple[str, ...]]:
    """Arguments of every `fix` whose style matches, e.g. 'nvt' or 'deform'."""
    found = []
    for command in script.all("fix"):
        if len(command.args) >= 3:
            candidate = command.args[2].lower()
            if candidate == style or candidate.startswith(f"{style}/"):
                found.append(list(command.args))
    return found


def _as_number(token: str | None) -> float | None:
    if token is None:
        return None
    try:
        return float(token)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# level 3 — task compliance
# --------------------------------------------------------------------------- #


def run_level3(script: ParsedScript, task: TaskSpec) -> list[CheckOutcome]:
    """Every level-3 check the task declares."""
    outcomes: list[CheckOutcome] = []
    for check in task.level3:
        outcomes.append(_level3_check(script, check))
    return outcomes


def _level3_check(script: ParsedScript, check: dict[str, Any]) -> CheckOutcome:
    kind = str(check.get("kind"))
    cid = str(check.get("id"))
    description = str(check.get("description", ""))

    if kind == "regex":
        found = re.search(str(check["pattern"]), script.text, re.MULTILINE) is not None
        return CheckOutcome(cid, found, description, kind)

    if kind == "regex_any":
        patterns = check.get("patterns") or []
        found = any(re.search(str(p), script.text, re.MULTILINE) for p in patterns)
        return CheckOutcome(cid, found, description, kind)

    if kind == "pattern_number":
        command = str(check.get("command", ""))
        index = int(check.get("argument", 0))
        expected = float(check.get("expected", 0.0))
        tolerance = float(check.get("tolerance", 0.0))
        args = _command_arguments(script, command)
        if args is None:
            return CheckOutcome(cid, False, f"no {command!r} command", kind)
        value = _as_number(args[index] if index < len(args) else None)
        if value is None:
            return CheckOutcome(cid, None, f"{command} argument {index} is not a number", kind,
                                needs_human=True)
        ok = abs(value - expected) <= tolerance
        return CheckOutcome(cid, ok, f"{command} = {value}, expected {expected}±{tolerance}", kind)

    if kind == "regex_value":
        # `regex_value` locates a named token in a fix's arguments and reads a
        # value at a fixed offset from it, which is how `sphere x y z r` is read.
        style = str(check.get("command", ""))
        token = str(check.get("token", ""))
        offset = int(check.get("value_index", 0))
        expected = float(check.get("expected", 0.0))
        tolerance = float(check.get("tolerance", 0.0))
        for args in _fix_arguments(script, style):
            if token in args:
                position = args.index(token)
                if position + offset < len(args):
                    value = _as_number(args[position + offset])
                    if value is not None:
                        ok = abs(value - expected) <= tolerance
                        return CheckOutcome(cid, ok, f"{style} {token} → {value}, expected {expected}", kind)
        return CheckOutcome(cid, False, f"no {style} command with a {token!r} argument", kind)

    if kind == "run_steps_min":
        minimum = int(check.get("minimum", 0))
        steps = [int(c.args[0]) for c in script.all("run") if c.args and c.args[0].isdigit()]
        if not steps:
            return CheckOutcome(cid, False, "no run with a literal step count", kind)
        longest = max(steps)
        return CheckOutcome(cid, longest >= minimum, f"longest run {longest}, need ≥ {minimum}", kind)

    if kind == "lattice_density":
        # `lattice fcc 0.8442` takes a reduced NUMBER DENSITY, not a lattice
        # constant — the single most common LJ error. Only the density form is
        # accepted, because the lattice-constant form silently builds a dilute
        # system that still runs.
        expected = float(check.get("expected", 0.0))
        tolerance = float(check.get("tolerance", 0.0))
        args = _command_arguments(script, "lattice")
        if args is None:
            return CheckOutcome(cid, False, "no lattice command", kind)
        value = _as_number(args[1] if len(args) > 1 else None)
        if value is None:
            return CheckOutcome(cid, None, "lattice scale is not a number", kind, needs_human=True)
        ok = abs(value - expected) <= tolerance
        return CheckOutcome(cid, ok, f"lattice scale {value}, expected {expected}±{tolerance}", kind)

    if kind == "fix_group_distinct":
        style = str(check.get("fix_command", ""))
        groups = [args[1] for args in _fix_arguments(script, style) if len(args) > 1]
        if not groups:
            return CheckOutcome(cid, False, f"no {style} fix", kind)
        # Applying a force fix to `all` pushes every atom, including the ones
        # that should be a substrate.
        ok = all(g != "all" for g in groups)
        return CheckOutcome(cid, ok, f"{style} acts on {groups}", kind)

    if kind == "damping_ratio":
        # Tdamp is a TIME constant, so it must scale with the timestep. A
        # literature value copied across unit systems silently stops the
        # thermostat acting at all.
        style = str(check.get("command", ""))
        index = int(check.get("argument", 0))
        minimum = float(check.get("minimum_ratio", 0.0))
        maximum = float(check.get("maximum_ratio", 1e18))
        timestep_args = _command_arguments(script, "timestep")
        dt = _as_number(timestep_args[0]) if timestep_args else None
        if dt is None or dt <= 0:
            return CheckOutcome(cid, False, "no usable timestep to compare against", kind)
        for args in _fix_arguments(script, style):
            damping = _as_number(args[index] if index < len(args) else None)
            if damping is None:
                continue
            ratio = damping / dt
            ok = minimum <= ratio <= maximum
            return CheckOutcome(cid, ok, f"damping/timestep = {ratio:.1f}, expected {minimum:g}–{maximum:g}", kind)
        return CheckOutcome(cid, False, f"no {style} fix with a numeric damping constant", kind)

    return CheckOutcome(cid, None, f"unimplemented check kind {kind!r}", kind, needs_human=True)


# --------------------------------------------------------------------------- #
# level 4 — physical sanity
# --------------------------------------------------------------------------- #


def parse_thermo(text: str) -> tuple[list[str], list[list[float]]]:
    """Extract the thermo table from a LAMMPS log, across every run in the script.

    LAMMPS reprints the header for each `run` command, so a normal staged script
    — equilibrate at one temperature, then ramp to another — leaves several
    tables in one log. Reading only the first describes the *equilibration*, not
    the experiment.

    That was not a cosmetic difference. `temperature_reached` samples the second
    half of the rows it is given, so on a two-stage melting run it measured the
    cold stage and reported "Temp left [0.6, 2] with 0.087" — failing a script
    that had done exactly what was asked. Only a multi-stage log executed
    end-to-end exposed it.

    Tables are merged in file order when their headers agree. A table with a
    different header is a different set of observables; merging rows of
    differing width would corrupt every index-based reader downstream, so it is
    left out rather than concatenated.
    """
    lines = text.splitlines()
    columns: list[str] = []
    rows: list[list[float]] = []

    for index, line in enumerate(lines):
        if not _THERMO_HEADER.match(line):
            continue
        header = line.split()
        table: list[list[float]] = []
        for candidate in lines[index + 1 :]:
            if not candidate.strip():
                break
            parts = candidate.split()
            try:
                table.append([float(p) for p in parts])
            except ValueError:
                break
        if not table:
            continue
        if not columns:
            columns, rows = header, table
        elif header == columns:
            rows.extend(table)

    return columns, rows


def run_level4(check_spec: tuple[dict[str, Any], ...], log_text: str) -> list[CheckOutcome]:
    """Every level-4 check, evaluated against the LAMMPS log."""
    columns, rows = parse_thermo(log_text)
    outcomes: list[CheckOutcome] = []
    for check in check_spec:
        outcomes.append(_level4_check(check, columns, rows, log_text))
    return outcomes


def _column(columns: list[str], name: str) -> int | None:
    for index, column in enumerate(columns):
        if column == name or column.startswith(name):
            return index
    return None


def _level4_check(
    check: dict[str, Any],
    columns: list[str],
    rows: list[list[float]],
    log_text: str,
) -> CheckOutcome:
    kind = str(check.get("kind"))
    cid = str(check.get("id"))
    description = str(check.get("description", ""))

    if kind == "human_review_required":
        # Reported, never scored. Marking a question as fine because it could not
        # be checked is the failure mode this exists to prevent.
        return CheckOutcome(cid, None, description or "needs human review", kind, needs_human=True)

    if kind == "log_forbidden":
        pattern = re.compile(str(check.get("pattern", "")))
        hits = [line for line in log_text.splitlines() if pattern.search(line) and line.startswith("ERROR")]
        if not log_text.strip():
            return CheckOutcome(cid, None, "no LAMMPS log to inspect", kind, needs_human=True)
        return CheckOutcome(cid, not hits, hits[0][:160] if hits else "no failure pattern", kind)

    if not rows:
        return CheckOutcome(cid, None, "no thermo table in the log", kind, needs_human=True)

    if kind == "log_range":
        name = str(check.get("column", ""))
        index = _column(columns, name)
        if index is None:
            return CheckOutcome(cid, None, f"column {name!r} is not in the thermo table", kind,
                                needs_human=True)
        fraction = float(check.get("after_fraction", 0.0))
        start = int(len(rows) * fraction)
        window = [row[index] for row in rows[start:] if index < len(row)]
        if not window:
            return CheckOutcome(cid, None, "no samples after the equilibration window", kind,
                                needs_human=True)
        low, high = float(check.get("min", float("-inf"))), float(check.get("max", float("inf")))
        worst = next((v for v in window if not (low <= v <= high)), None)
        ok = worst is None
        return CheckOutcome(cid, ok, f"{name} stayed in [{low:g}, {high:g}]" if ok
                            else f"{name} left [{low:g}, {high:g}] with {worst:g}", kind)

    if kind == "log_finite":
        names = check.get("columns") or []
        for name in names:
            index = _column(columns, str(name))
            if index is None:
                return CheckOutcome(cid, None, f"column {name!r} is absent", kind, needs_human=True)
            for row in rows:
                if index < len(row):
                    value = row[index]
                    if value != value or value in (float("inf"), float("-inf")):
                        return CheckOutcome(cid, False, f"{name} became {value}", kind)
        return CheckOutcome(cid, True, f"{', '.join(map(str, names))} all finite", kind)

    if kind == "log_series_increasing":
        name = str(check.get("column", ""))
        index = _column(columns, name)
        if index is None:
            return CheckOutcome(cid, None, f"column {name!r} is not in the thermo table", kind,
                                needs_human=True)
        series = [row[index] for row in rows if index < len(row)]
        if len(series) < 2:
            return CheckOutcome(cid, None, "too few samples to judge growth", kind, needs_human=True)
        growth = series[-1] / series[0] if series[0] else float("inf")
        need = float(check.get("min_growth_factor", 2.0))
        ok = growth >= need
        return CheckOutcome(cid, ok, f"{name} grew {growth:.2f}×, need ≥ {need:g}×", kind)

    if kind == "log_stability":
        name = str(check.get("column", ""))
        index = _column(columns, name)
        if index is None:
            return CheckOutcome(cid, None, f"column {name!r} is absent", kind, needs_human=True)
        series = [row[index] for row in rows if index < len(row)]
        if not series or series[0] == 0:
            return CheckOutcome(cid, None, "no usable baseline", kind, needs_human=True)
        drift = abs(series[-1] - series[0]) / abs(series[0])
        limit = float(check.get("max_relative_drift", 0.5))
        return CheckOutcome(cid, drift <= limit, f"{name} drifted {drift:.1%}, limit {limit:.0%}", kind)

    return CheckOutcome(cid, None, f"unimplemented check kind {kind!r}", kind, needs_human=True)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


#: Ordered most-specific-first. The first match names the failure, so a missing
#: `units` is reported as a knowledge error rather than a runtime one.
_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ("UNITS_MISSING", "knowledge_error"),
    ("TIMESTEP_MISSING", "knowledge_error"),
    ("TIMESTEP_NONPOSITIVE", "bad_parameter"),
    ("ATOM_STYLE_MISSING", "knowledge_error"),
    ("STRUCTURE_NOT_INITIALISED", "missing_command"),
    ("PAIR_STYLE_MISSING", "missing_command"),
    ("PAIR_COEFF_MISSING", "missing_command"),
    ("ORDER_VIOLATION", "command_order_error"),
    ("TASK_REQUIRED_COMMAND_MISSING", "missing_command"),
    ("TASK_REQUIRED_PATTERN_MISSING", "task_noncompliance"),
    ("FILE_MISSING_REQUIRED", "missing_file"),
    ("POTENTIAL_FILE_MISSING", "invalid_reference"),
    ("REFERENCED_FILE_MISSING", "invalid_reference"),
    ("CONFLICT_MULTIPLE_INTEGRATORS", "wrong_ensemble"),
    ("CONFLICT_DEFORM_AND_BAROSTAT_SAME_AXIS", "bad_parameter"),
)


def categorise(level1_codes: tuple[str, ...], level2_ran: bool | None) -> str:
    """Name the failure category for the taxonomy the brief requires.

    Ordered, because a run can fail several checks at once and the first, most
    specific cause is what a reader needs. A structural failure outranks a
    runtime one: fixing the runtime symptom of a missing `units` command would
    not help.
    """
    for code, category in _CATEGORY_RULES:
        if code in level1_codes:
            return category
    if level2_ran is False:
        return "runtime_error"
    if level2_ran is None:
        return "unknown"
    return "none"


def evaluate_run(
    workspace: Path | str,
    task: TaskSpec,
    *,
    run_level2: bool = True,
    timeout: int = 900,
) -> Evaluation:
    """Score one run at all four levels.

    Levels 1 and 2 reuse the validator and the local runner rather than
    reimplementing them. Scoring a run against checks the agent was never gated
    on would make the ablation uninterpretable, so there is exactly one
    implementation of each.
    """
    from adapter.runner import run_local
    from adapter.validator import validate_workspace

    workspace = Path(workspace)
    evaluation = Evaluation(task_id=task.id)

    result = validate_workspace(
        workspace,
        task=task,
        supported_atom_styles=(),
        supported_unit_styles=(),
    )
    evaluation.level1_valid = result.valid
    evaluation.level1_findings = result.to_dict()

    script_path = next((p for p in sorted(workspace.glob("in.*")) if p.is_file()), None)
    script = parse_script(
        script_path.read_text(encoding="utf-8", errors="replace") if script_path else ""
    )
    evaluation.level3 = run_level3(script, task)

    if run_level2 and script_path is not None:
        outcome = run_local(workspace, timeout=timeout)
        evaluation.level2_ran = outcome.ran
        evaluation.level2_detail = outcome.error
        log_text = (
            outcome.log_path.read_text(encoding="utf-8", errors="replace")
            if outcome.log_path and outcome.log_path.is_file()
            else ""
        )
        evaluation.level4 = run_level4(task.level4, log_text)
    else:
        evaluation.level4 = [
            CheckOutcome(str(c.get("id")), None, "not run", str(c.get("kind")), needs_human=True)
            for c in task.level4
        ]

    evaluation.failure_category = categorise(result.failure_codes, evaluation.level2_ran)
    return evaluation
