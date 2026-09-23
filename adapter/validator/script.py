"""Parse a LAMMPS input script into an ordered command list.

Deliberately structural, not semantic: it answers "which commands, in what
order, with what arguments, on which lines" and stops there. Every judgement
about whether an order is *correct* belongs to the rules, so that a rule can be
read, tested and argued with on its own.

Two honesty features matter more than they look:

* **Control flow is detected and reported.** ``if``, ``jump``, ``include`` and
  friends mean the executed order is not the written order. A validator that
  silently assumed top-to-bottom execution would produce confident nonsense on
  such a script, so the parse records the fact and the ordering rules downgrade
  themselves to warnings.
* **Variables are recognised as opaque.** ``${x}`` in an argument may expand to a
  filename, a number, or a command fragment. The parser records the reference
  rather than pretending to know its value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

__all__ = ["Command", "ParsedScript", "parse_script", "strip_comment", "VARIABLE_REFERENCE"]

#: `${name}`, `$(expr)`, or `$x` — a value only known at runtime.
VARIABLE_REFERENCE = re.compile(r"\$\{[^}]*\}|\$\([^)]*\)|\$[A-Za-z_][A-Za-z0-9_]*")

#: Commands that make the executed order differ from the written order.
CONTROL_FLOW_KEYWORDS = frozenset(
    {"if", "jump", "include", "label", "next", "while", "else", "quit"}
)

#: Commands that define state rather than act on the system. Used by rules that
#: care about declaration-before-use.
DECLARATION_KEYWORDS = frozenset(
    {
        "units",
        "atom_style",
        "boundary",
        "lattice",
        "region",
        "create_box",
        "create_atoms",
        "read_data",
        "read_restart",
        "mass",
        "pair_style",
        "bond_style",
        "angle_style",
        "dihedral_style",
        "improper_style",
        "kspace_style",
        "compute",
        "variable",
        "group",
        "set",
    }
)


@dataclass(frozen=True)
class Command:
    """One parsed command."""

    line: int
    keyword: str
    """Lowercased, for matching. LAMMPS command names are written lowercase."""
    raw_keyword: str
    args: tuple[str, ...]
    raw: str

    @property
    def argument_string(self) -> str:
        return " ".join(self.args)

    def arg(self, index: int) -> str | None:
        """The n-th argument, or ``None`` when absent."""
        return self.args[index] if 0 <= index < len(self.args) else None

    @property
    def has_variable(self) -> bool:
        return bool(VARIABLE_REFERENCE.search(self.raw))

    def __str__(self) -> str:
        return f"line {self.line}: {self.raw_keyword}"


@dataclass
class ParsedScript:
    """A parsed script plus the structural facts rules need."""

    commands: list[Command] = field(default_factory=list)
    #: Command keywords that make written order differ from executed order.
    control_flow: set[str] = field(default_factory=set)
    #: Names defined by `variable`, with the style used.
    variables: dict[str, str] = field(default_factory=dict)
    path: Path | None = None
    text: str = ""

    @property
    def has_control_flow(self) -> bool:
        return bool(self.control_flow)

    def keywords(self) -> list[str]:
        return [c.keyword for c in self.commands]

    def count(self, keyword: str) -> int:
        return sum(1 for c in self.commands if c.keyword == keyword)

    def all(self, keyword: str) -> list[Command]:
        return [c for c in self.commands if c.keyword == keyword]

    def first(self, keyword: str) -> Command | None:
        for command in self.commands:
            if command.keyword == keyword:
                return command
        return None

    def first_index(self, keyword: str) -> int | None:
        for index, command in enumerate(self.commands):
            if command.keyword == keyword:
                return index
        return None

    def present(self, *keywords: str) -> bool:
        """Whether any of *keywords* appears."""
        return any(self.first(k) is not None for k in keywords)

    def __iter__(self) -> Iterator[Command]:
        return iter(self.commands)

    def __len__(self) -> int:
        return len(self.commands)


def strip_comment(line: str) -> str:
    """Remove a trailing ``#`` comment.

    LAMMPS has no string-literal syntax, so a ``#`` always begins a comment.
    """
    index = line.find("#")
    return line if index < 0 else line[:index]


def parse_script(text: str, path: Path | None = None) -> ParsedScript:
    """Parse LAMMPS input *text*.

    Blank lines, comments, and ``&``-continued lines are handled. A line whose
    content is only a variable reference is skipped: it is a variable expansion
    used as a command, whose meaning is not statically known.
    """
    script = ParsedScript(path=path, text=text)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    index = 0
    while index < len(lines):
        start_line = index + 1
        logical = strip_comment(lines[index]).rstrip()
        # Join `&` continuations. The trailing `&` may follow a comment.
        while logical.rstrip().endswith("&") and index + 1 < len(lines):
            logical = logical.rstrip()[:-1].rstrip()
            index += 1
            logical = f"{logical} {strip_comment(lines[index]).strip()}".strip()
        index += 1

        if not logical.strip():
            continue

        parts = logical.split()
        raw_keyword = parts[0]
        keyword = raw_keyword.lower()

        # A bare ${var} as a whole command is an expansion whose meaning is
        # unknown; recording it as a keyword would be a lie.
        if VARIABLE_REFERENCE.fullmatch(raw_keyword):
            continue

        command = Command(
            line=start_line,
            keyword=keyword,
            raw_keyword=raw_keyword,
            args=tuple(parts[1:]),
            raw=logical.strip(),
        )
        script.commands.append(command)

        if keyword in CONTROL_FLOW_KEYWORDS:
            script.control_flow.add(keyword)
        if keyword == "variable" and len(command.args) >= 2:
            script.variables[command.args[0]] = command.args[1]

    return script


def parse_script_file(path: Path | str) -> ParsedScript:
    """Parse a script from disk.

    Raises:
        FileNotFoundError: the path does not exist.
        OSError: the file cannot be read.
    """
    path = Path(path)
    return parse_script(path.read_text(encoding="utf-8", errors="replace"), path=path)


def unique_keywords(script: ParsedScript) -> dict[str, list[Command]]:
    """Every command keyword mapped to its occurrences, for summary reporting."""
    out: dict[str, list[Command]] = {}
    for command in script.commands:
        out.setdefault(command.keyword, []).append(command)
    return out


def summarise(script: ParsedScript) -> dict[str, Any]:
    """Compact facts about a script, safe to embed in a tool result."""
    return {
        "commands": len(script.commands),
        "distinct_keywords": len(unique_keywords(script)),
        "control_flow": sorted(script.control_flow),
        "variables_defined": len(script.variables),
        "lines": len(script.text.split("\n")),
    }
