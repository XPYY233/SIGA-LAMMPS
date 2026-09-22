"""Deterministic validation rules.

**No LLM participates here.** Every finding is produced by comparing parsed
structure against a stated rule, which is what makes the result reproducible,
auditable, and safe to gate termination on.

Design stance on false positives: a rule that rejects a valid-but-unusual script
is worse than one that misses a defect. The design principles forbid treating a
reference implementation as the only correct answer, so nothing here compares
against a reference. Rules emit an **error** only when the script cannot work as
written, a **warning** when a careful author would reconsider, and a
**suggestion** when it is merely worth knowing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from adapter.tasks import TaskSpec
from adapter.validator.findings import ERROR, SUGGESTION, WARNING, Finding
from adapter.validator.script import Command, ParsedScript, VARIABLE_REFERENCE

__all__ = ["RuleContext", "run_rules"]

#: Fixes that integrate motion. Two on one group double-integrate.
INTEGRATOR_FIXES = (
    "nve",
    "nvt",
    "npt",
    "nph",
    "nve/limit",
    "nvt/sllod",
    "npt/sllod",
    "nvt/asphere",
    "nvt/sphere",
    "nvt/body",
    "brownian",
    "langevin",
    "dpd",
    "rigid",
    "rigid/nve",
    "rigid/nvt",
)

#: `pair_style` values whose `pair_coeff` must name a potential file.
FILE_BASED_PAIR_STYLES = (
    "eam",
    "eam/alloy",
    "eam/fs",
    "eam/he",
    "meam",
    "meam/spline",
    "tersoff",
    "tersoff/zbl",
    "sw",
    "stillinger-weber",
    "airebo",
    "rebo",
    "bop",
    "adp",
    "comb",
    "comb3",
    "vashishta",
    "buck/coul/long/cs",
    "gw",
    "gw/zbl",
    "mliap",
    "pace",
    "pod",
    "kim",
    "nb3b/harmonic",
)

#: Commands that read a file from the working directory.
FILE_READING_COMMANDS = frozenset(
    {"read_data", "read_restart", "read_dump", "include", "molecule", "potential"}
)

#: Command arguments that are files only in some positions; handled per command.
_EXCLUDED_FILE_TOKENS = frozenset(
    {"NULL", "none", "yes", "no", "on", "off", "true", "false", "*", "all"}
)


@dataclass
class RuleContext:
    """Everything the rules need, gathered once."""

    script: ParsedScript
    workspace: Path | None = None
    task: TaskSpec | None = None
    supported_atom_styles: tuple[str, ...] = ()
    supported_unit_styles: tuple[str, ...] = ()
    universal_required: tuple[str, ...] = ()
    max_lines: int = 400
    findings: list[Finding] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    # -- helpers ------------------------------------------------------------ #

    def add(
        self,
        code: str,
        message: str,
        severity: str,
        line: int | None = None,
        **evidence: Any,
    ) -> None:
        self.findings.append(
            Finding(code=code, message=message, severity=severity, line=line, evidence=evidence)
        )

    def error(self, code: str, message: str, line: int | None = None, **evidence: Any) -> None:
        self.add(code, message, ERROR, line, **evidence)

    def warn(self, code: str, message: str, line: int | None = None, **evidence: Any) -> None:
        self.add(code, message, WARNING, line, **evidence)

    def suggest(self, code: str, message: str, line: int | None = None, **evidence: Any) -> None:
        self.add(code, message, SUGGESTION, line, **evidence)

    @property
    def control_flow(self) -> bool:
        return self.script.has_control_flow

    def exists(self, name: str, *, line: int | None = None) -> bool | None:
        """Whether *name* resolves to a file in the workspace.

        Returns ``None`` when the question cannot be answered — no workspace was
        given, or the name contains a variable reference. Callers must not treat
        unknown as failure, which is why this is tri-state rather than a bool.
        """
        if self.workspace is None:
            return None
        if VARIABLE_REFERENCE.search(name):
            return None
        candidate = Path(name)
        if candidate.is_absolute():
            return candidate.exists()
        return (self.workspace / candidate).exists()


# --------------------------------------------------------------------------- #
# 1-2: files and ordering
# --------------------------------------------------------------------------- #


def rule_required_files(ctx: RuleContext) -> None:
    """Required files exist (brief check 1)."""
    if ctx.task is None or ctx.workspace is None:
        return
    for pattern in ctx.task.required_files:
        if not any(ctx.workspace.glob(pattern)):
            ctx.error(
                "FILE_MISSING_REQUIRED",
                f"the task requires a file matching {pattern!r}; none exists in the workspace",
                pattern=pattern,
            )


_ORDER_RULES: tuple[tuple[str, str, str], ...] = (
    ("units", "lattice", "units must be set before any command that carries a physical quantity"),
    ("units", "region", "units must be set before region geometry is interpreted"),
    ("units", "pair_style", "units must be set before a force field is defined"),
    ("atom_style", "lattice", "atom_style determines which per-atom fields exist"),
    ("atom_style", "create_box", "atom_style must precede create_box"),
    ("atom_style", "read_data", "atom_style must precede read_data"),
    ("lattice", "region", "lattice must precede region, because region units default to lattice"),
    ("lattice", "create_box", "lattice must precede create_box"),
    ("region", "create_box", "create_box needs a region that already exists"),
    ("create_box", "create_atoms", "create_atoms needs a box that already exists"),
    ("pair_style", "pair_coeff", "pair_coeff configures a pair_style that must already be set"),
    ("pair_style", "run", "a run needs a force field"),
    ("mass", "run", "atoms need masses before dynamics"),
)


def rule_command_ordering(ctx: RuleContext) -> None:
    """Declare-before-use ordering (brief checks 2 and 14)."""
    script = ctx.script
    # Written order is not executed order when control flow is present, so
    # ordering findings become advisory rather than assertions.
    severity = WARNING if ctx.control_flow else ERROR
    order_code = "ORDER_VIOLATION_CONTROL_FLOW" if ctx.control_flow else "ORDER_VIOLATION"

    for earlier, later, explanation in _ORDER_RULES:
        first_earlier = script.first(earlier)
        first_later = script.first(later)
        if first_earlier is None or first_later is None:
            continue
        if first_earlier.line > first_later.line:
            message = (
                f"{later!r} (line {first_later.line}) appears before {earlier!r} "
                f"(line {first_earlier.line}): {explanation}"
            )
            if ctx.control_flow:
                message += (
                    " — note this script contains control flow, so the written order "
                    "may not be the executed order"
                )
            ctx.add(order_code, message, severity, first_later.line, earlier=earlier, later=later)

    # read_data is exclusive with the lattice-construction family.
    read_data = script.first("read_data")
    for keyword in ("create_box", "lattice", "create_atoms", "region"):
        building = script.first(keyword)
        if read_data is not None and building is not None:
            ctx.error(
                "CONFLICT_READ_DATA_WITH_CONSTRUCTION",
                f"{keyword!r} (line {building.line}) is used alongside read_data "
                f"(line {read_data.line}); read_data defines the system, so the "
                "lattice-construction commands are redundant or wrong",
                line=max(read_data.line, building.line),
                keyword=keyword,
            )


# --------------------------------------------------------------------------- #
# 3-7: units, style, structure, force field
# --------------------------------------------------------------------------- #


def rule_units(ctx: RuleContext) -> None:
    """Units present, recognised, and consistent with what was asked for (check 3)."""
    command = ctx.script.first("units")
    if command is None:
        ctx.error("UNITS_MISSING", "no 'units' command; every quantity is undefined without it")
        return
    style = command.arg(0)
    if style is None:
        ctx.error("UNITS_NO_STYLE", "'units' given with no style", command.line)
        return
    ctx.facts["units"] = style
    if ctx.supported_unit_styles and style not in ctx.supported_unit_styles:
        ctx.warn(
            "UNITS_UNSUPPORTED",
            f"units style {style!r} is outside the styles this validator is confident about",
            command.line,
            style=style,
        )
    if command.line != ctx.script.commands[0].line:
        ctx.warn(
            "UNITS_NOT_FIRST",
            f"'units' is not the first command (line {command.line}); it fixes the "
            "meaning of every number that follows",
            command.line,
        )


def rule_atom_style(ctx: RuleContext) -> None:
    """atom_style present and consistent (check 4)."""
    command = ctx.script.first("atom_style")
    if command is None:
        # Only a problem if atoms are created or read without a style.
        if ctx.script.present("create_box", "create_atoms", "read_data"):
            ctx.error(
                "ATOM_STYLE_MISSING",
                "atoms are created or read but no 'atom_style' is set",
            )
        return
    style = command.arg(0)
    if style is None:
        ctx.error("ATOM_STYLE_NO_STYLE", "'atom_style' given with no style", command.line)
        return
    ctx.facts["atom_style"] = style
    if ctx.supported_atom_styles and style not in ctx.supported_atom_styles:
        ctx.warn(
            "ATOM_STYLE_UNSUPPORTED",
            f"atom_style {style!r} is outside the styles this validator is confident about",
            command.line,
            style=style,
        )


def rule_structure_initialised(ctx: RuleContext) -> None:
    """The system is initialised one way or the other (check 5)."""
    script = ctx.script
    if not script.present("run", "minimize"):
        return  # a setup-only fragment is not required to build a system

    if script.present("read_data", "read_restart"):
        return

    missing = [k for k in ("lattice", "region", "create_box", "create_atoms") if not script.present(k)]
    if missing:
        ctx.error(
            "STRUCTURE_NOT_INITIALISED",
            "the simulation runs but the system is never created: no read_data and no "
            f"{'/'.join(missing)} command(s)",
            missing=missing,
        )


def rule_force_field(ctx: RuleContext) -> None:
    """pair_style / pair_coeff present and coherent (checks 6 and 7)."""
    script = ctx.script
    if not script.present("run", "minimize"):
        return
    style_cmd = script.first("pair_style")
    if style_cmd is None:
        ctx.error("PAIR_STYLE_MISSING", "no 'pair_style'; a run needs a force field")
        return
    style = style_cmd.arg(0)
    ctx.facts["pair_style"] = style
    coeffs = script.all("pair_coeff")
    if not coeffs:
        ctx.error(
            "PAIR_COEFF_MISSING",
            f"pair_style {style!r} is set but no 'pair_coeff' follows",
            style_cmd.line,
            style=style,
        )
        return

    if style in FILE_BASED_PAIR_STYLES:
        _check_potential_file(ctx, style, coeffs)


def _check_potential_file(ctx: RuleContext, style: str, coeffs: list[Command]) -> None:
    """A file-based potential must name a file that exists (check 7 / 12)."""
    for command in coeffs:
        # `pair_coeff i j <file> [elements]` — the file is the first token that
        # is not a type index and not a wildcard.
        candidate = None
        for token in command.args:
            if token in _EXCLUDED_FILE_TOKENS or token.isdigit() or "*" in token:
                continue
            candidate = token
            break
        if candidate is None:
            ctx.error(
                "POTENTIAL_FILE_NOT_SPECIFIED",
                f"pair_style {style!r} requires a potential file but pair_coeff names none",
                command.line,
                style=style,
            )
            return
        exists = ctx.exists(candidate, line=command.line)
        if exists is False:
            ctx.error(
                "POTENTIAL_FILE_MISSING",
                f"potential file {candidate!r} referenced at line {command.line} does not "
                "exist in the workspace",
                command.line,
                file=candidate,
            )
        elif exists is None:
            ctx.suggest(
                "POTENTIAL_FILE_UNVERIFIED",
                f"could not check that potential file {candidate!r} exists "
                "(no workspace, or the name contains a variable)",
                command.line,
                file=candidate,
            )


# --------------------------------------------------------------------------- #
# 8-10: ensemble, timestep, run
# --------------------------------------------------------------------------- #


def rule_ensemble(ctx: RuleContext) -> None:
    """Integrator present, and not doubled (checks 8 and 13)."""
    script = ctx.script
    if not script.present("run"):
        return

    integrators: list[tuple[Command, str, str]] = []
    for command in script.all("fix"):
        if len(command.args) < 3:
            continue
        # fix <id> <group> <style> ...
        fix_id, group, style = command.args[0], command.args[1], command.args[2].lower()
        if style in INTEGRATOR_FIXES or any(style.startswith(f"{i}/") for i in INTEGRATOR_FIXES):
            integrators.append((command, group, style))

    if not integrators:
        ctx.warn(
            "ENSEMBLE_MISSING",
            "a run is requested but no integrating fix (nve/nvt/npt/...) is defined; "
            "the system will not move",
        )
        return

    ctx.facts["integrators"] = [style for _, _, style in integrators]

    # Two integrators on the same group double-integrate.
    by_group: dict[str, list[tuple[Command, str]]] = {}
    for command, group, style in integrators:
        by_group.setdefault(group, []).append((command, style))
    for group, entries in by_group.items():
        if len(entries) > 1:
            styles = ", ".join(f"{s!r} (line {c.line})" for c, s in entries)
            ctx.error(
                "CONFLICT_MULTIPLE_INTEGRATORS",
                f"group {group!r} has more than one integrator: {styles}. Only one fix may "
                "integrate a group at a time; 'unfix' the previous one first",
                line=entries[1][0].line,
                group=group,
                styles=[s for _, s in entries],
            )

    # Redefining a fix id without unfixing fails at runtime.
    defined: dict[str, Command] = {}
    for command in script.all("fix"):
        if not command.args:
            continue
        fix_id = command.args[0]
        if fix_id in defined:
            ctx.warn(
                "FIX_ID_REDEFINED_WITHOUT_UNFIX",
                f"fix id {fix_id!r} is defined again at line {command.line} "
                f"(first at line {defined[fix_id].line}) without an intervening 'unfix'",
                command.line,
                fix_id=fix_id,
            )
        defined[fix_id] = command


def rule_timestep(ctx: RuleContext) -> None:
    """timestep present and positive (check 9)."""
    if not ctx.script.present("run", "minimize"):
        return
    command = ctx.script.first("timestep")
    if command is None:
        ctx.error(
            "TIMESTEP_MISSING",
            "no 'timestep' command. It defaults to 0.0, which does not merely change the "
            "trajectory — it makes the dynamics meaningless",
        )
        return
    raw = command.arg(0)
    if raw is None:
        ctx.error("TIMESTEP_NO_VALUE", "'timestep' given with no value", command.line)
        return
    try:
        value = float(raw)
    except ValueError:
        if VARIABLE_REFERENCE.search(raw):
            ctx.suggest(
                "TIMESTEP_UNVERIFIED",
                f"timestep {raw!r} is a variable; its value cannot be checked statically",
                command.line,
            )
            return
        ctx.error("TIMESTEP_NOT_A_NUMBER", f"timestep {raw!r} is not a number", command.line)
        return

    ctx.facts["timestep"] = value
    if value <= 0.0:
        ctx.error(
            "TIMESTEP_NONPOSITIVE",
            f"timestep is {value}; it must be positive for dynamics to advance",
            command.line,
            timestep=value,
        )
        return

    units = ctx.facts.get("units")
    bounds = {"lj": (1e-4, 0.02), "metal": (1e-5, 0.01), "real": (0.1, 5.0)}
    if units in bounds:
        low, high = bounds[units]
        if value > high:
            ctx.warn(
                "TIMESTEP_TOO_LARGE",
                f"timestep {value} is large for units {units!r} (a safe range is roughly "
                f"{low}–{high}); expect lost atoms or unstable dynamics",
                command.line,
                timestep=value,
                units=units,
            )


def rule_run(ctx: RuleContext) -> None:
    """A run is requested, with a sensible length (check 10)."""
    runs = ctx.script.all("run")
    if not runs and not ctx.script.present("minimize"):
        ctx.error("RUN_MISSING", "the script never requests a run or a minimize")
        return
    ctx.facts["run_steps"] = []
    for command in runs:
        raw = command.arg(0)
        if raw is None:
            ctx.error("RUN_NO_VALUE", "'run' given with no step count", command.line)
            continue
        try:
            steps = int(raw)
        except ValueError:
            if VARIABLE_REFERENCE.search(raw):
                ctx.suggest(
                    "RUN_UNVERIFIED",
                    f"run length {raw!r} is a variable; it cannot be checked statically",
                    command.line,
                )
                continue
            ctx.error("RUN_NOT_A_NUMBER", f"run length {raw!r} is not an integer", command.line)
            continue
        ctx.facts["run_steps"].append(steps)
        if steps == 0:
            ctx.suggest(
                "RUN_ZERO_STEPS",
                "run 0 performs setup only; correct for an initialisation check, but it "
                "produces no dynamics",
                command.line,
            )
        elif steps < 0:
            ctx.error("RUN_NEGATIVE", f"run length {steps} is negative", command.line)


# --------------------------------------------------------------------------- #
# 11-13: task requirements, file references, conflicts
# --------------------------------------------------------------------------- #


def rule_task_requirements(ctx: RuleContext) -> None:
    """The task's required commands and patterns are present (check 11)."""
    if ctx.task is None:
        return
    script = ctx.script

    for command in ctx.task.level1.required_commands:
        if not script.present(command):
            ctx.error(
                "TASK_REQUIRED_COMMAND_MISSING",
                f"the task requires a {command!r} command, which is absent",
                command=command,
                task=ctx.task.id,
            )

    text = script.text
    for entry in ctx.task.level1.required_patterns:
        try:
            pattern = re.compile(entry["pattern"], re.MULTILINE)
        except re.error:  # pragma: no cover - task files are validated on load
            continue
        if not pattern.search(text):
            ctx.error(
                "TASK_REQUIRED_PATTERN_MISSING",
                f"{entry.get('description') or 'a required form is absent'}"
                f" (expected a line matching {entry['pattern']!r})",
                pattern_id=entry["id"],
                task=ctx.task.id,
            )


def rule_file_references(ctx: RuleContext) -> None:
    """Every referenced file exists (check 12)."""
    script = ctx.script
    for command in script.commands:
        if command.keyword not in FILE_READING_COMMANDS:
            continue
        if not command.args:
            continue
        if command.keyword == "potential":
            candidates = command.args[:1]
        elif command.keyword == "molecule":
            candidates = command.args[1:2]
        else:
            candidates = command.args[:1]
        for name in candidates:
            if name in _EXCLUDED_FILE_TOKENS:
                continue
            exists = ctx.exists(name, line=command.line)
            if exists is False:
                ctx.error(
                    "REFERENCED_FILE_MISSING",
                    f"{command.keyword!r} at line {command.line} reads {name!r}, which does "
                    "not exist in the workspace",
                    command.line,
                    file=name,
                    command=command.keyword,
                )


def fixes_with_style(script: ParsedScript, *prefixes: str) -> list[tuple[Command, str]]:
    """Every ``fix`` whose style matches one of *prefixes*.

    Styles are the **third argument** of a ``fix`` command, not command
    keywords. So ``script.first("deform")`` finds nothing at all: the lookup
    silently matched no command and left the whole deform/barostat rule dead.
    This helper exists so that mistake cannot be repeated.
    """
    found: list[tuple[Command, str]] = []
    for command in script.all("fix"):
        if len(command.args) < 3:
            continue
        style = command.args[2].lower()
        if any(style == p or style.startswith(f"{p}/") for p in prefixes):
            found.append((command, style))
    return found


def rule_conflicts(ctx: RuleContext) -> None:
    """Obvious conflicts a script cannot satisfy simultaneously (check 13)."""
    script = ctx.script

    # fix deform plus a barostat on the same axis double-counts the strain.
    for deform, _ in fixes_with_style(script, "deform"):
        deform_axes = set(re.findall(r"\b([xyz])\b", " ".join(deform.args[3:]).lower()))
        for command, style in fixes_with_style(script, "npt", "nph"):
            barostat_axes = set(re.findall(r"\b([xyz])\s+0", " ".join(command.args[3:]).lower()))
            for axis in sorted(deform_axes & barostat_axes):
                ctx.error(
                    "CONFLICT_DEFORM_AND_BAROSTAT_SAME_AXIS",
                    f"fix deform acts on {axis} while {style!r} at line {command.line} also "
                    f"controls {axis}; the strain is applied twice. Couple the barostat "
                    "only to the axes you are not deforming",
                    line=command.line,
                    axis=axis,
                    barostat=style,
                    deform_line=deform.line,
                )

    # Thermostatting a group that is also held rigid is contradictory.
    rigid_groups = {
        command.args[1] for command in script.all("fix") if len(command.args) >= 4 and command.args[2].lower() == "rigid"
    }
    for command in script.all("fix"):
        if len(command.args) < 3:
            continue
        if command.args[2].lower().startswith(("nvt", "langevin", "temp")):
            if command.args[1] in rigid_groups:
                ctx.warn(
                    "CONFLICT_THERMOSTAT_ON_RIGID_GROUP",
                    f"group {command.args[1]!r} is held rigid and also thermostatted at "
                    f"line {command.line}; the thermostat cannot act on frozen degrees of freedom",
                    command.line,
                )

    # A compute referenced before it is defined.
    defined_computes = {c.args[0] for c in script.all("compute") if c.args}
    compute_definition_line = {
        c.args[0]: c.line for c in script.all("compute") if c.args
    }
    for command in script.commands:
        for match in re.finditer(r"\bc_([A-Za-z0-9_]+)", command.raw):
            name = match.group(1)
            if name in defined_computes and compute_definition_line[name] > command.line:
                ctx.error(
                    "COMPUTE_USED_BEFORE_DEFINED",
                    f"compute {name!r} is referenced at line {command.line} but defined at "
                    f"line {compute_definition_line[name]}",
                    command.line,
                    compute=name,
                )


def rule_script_size(ctx: RuleContext) -> None:
    """Guard against a script that is not a benchmark script at all."""
    lines = len(ctx.script.text.split("\n"))
    if ctx.max_lines and lines > ctx.max_lines:
        ctx.warn(
            "SCRIPT_TOO_LONG",
            f"the script has {lines} lines, beyond the {ctx.max_lines}-line expectation",
            lines=lines,
        )


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

_RULES = (
    rule_script_size,
    rule_required_files,
    rule_units,
    rule_atom_style,
    rule_command_ordering,
    rule_structure_initialised,
    rule_force_field,
    rule_ensemble,
    rule_timestep,
    rule_run,
    rule_task_requirements,
    rule_file_references,
    rule_conflicts,
)


def run_rules(ctx: RuleContext) -> RuleContext:
    """Run every rule in a fixed order.

    Deterministic order matters: findings are reported in a stable sequence so
    that two runs over the same script produce byte-identical output, which the
    benchmark compares.
    """
    for rule in _RULES:
        rule(ctx)
    return ctx


def iter_rule_names() -> Iterable[str]:
    return (rule.__name__ for rule in _RULES)
