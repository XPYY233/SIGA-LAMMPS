"""Structured validation findings.

The result shape is fixed by the project brief::

    {"valid": false, "errors": [...], "warnings": [...], "suggestions": [...]}

Every finding carries a stable ``code`` and, where possible, the offending line
and machine-readable ``evidence``. That is deliberate: the benchmark buckets
failures by category, S formats a repair instruction, and a human reads the
report — three consumers that cannot all be served by a prose string.

Codes are stable identifiers, not messages. Renaming one is a breaking change to
the benchmark's failure taxonomy; changing a message is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Finding", "Severity", "ValidationResult"]

Severity = str  # "error" | "warning" | "suggestion"

ERROR: Severity = "error"
WARNING: Severity = "warning"
SUGGESTION: Severity = "suggestion"


@dataclass(frozen=True)
class Finding:
    """One validation finding."""

    code: str
    message: str
    severity: Severity
    line: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.line is not None:
            out["line"] = self.line
        if self.evidence:
            out["evidence"] = self.evidence
        return out


@dataclass(frozen=True)
class ValidationResult:
    """The complete outcome of one validation run."""

    findings: tuple[Finding, ...] = ()
    task_id: str | None = None
    script: str | None = None
    #: Facts the caller may want to display. Not part of the pass/fail decision.
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == WARNING)

    @property
    def suggestions(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == SUGGESTION)

    @property
    def valid(self) -> bool:
        """Whether the script passes.

        Only errors decide this. A warning is something a careful author would
        reconsider; blocking on one would reject valid-but-unusual scripts, which
        the design principles explicitly forbid.
        """
        return not self.errors

    @property
    def failure_codes(self) -> tuple[str, ...]:
        """Stable codes of the errors, for the benchmark's failure taxonomy."""
        return tuple(f.code for f in self.errors)

    def to_dict(self) -> dict[str, Any]:
        """The contract S and the MCP tool both emit."""
        return {
            "valid": self.valid,
            "errors": [f.to_dict() for f in self.errors],
            "warnings": [f.to_dict() for f in self.warnings],
            "suggestions": [f.to_dict() for f in self.suggestions],
            "task": self.task_id,
            "counts": {
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "suggestions": len(self.suggestions),
            },
            "facts": self.facts,
        }

    def render(self) -> str:
        """A compact human- and model-readable summary."""
        lines: list[str] = []
        if self.valid and not self.warnings:
            lines.append("validation passed with no findings")
        for label, group in (("ERROR", self.errors), ("WARNING", self.warnings), ("NOTE", self.suggestions)):
            for finding in group:
                location = f" (line {finding.line})" if finding.line else ""
                lines.append(f"{label} [{finding.code}]{location}: {finding.message}")
        return "\n".join(lines) if lines else "validation produced no findings"
