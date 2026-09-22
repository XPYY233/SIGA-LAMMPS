"""X — deterministic LAMMPS input validation.

No LLM participates. Every finding compares parsed script structure against a
stated rule, which is what makes the result reproducible, auditable, and safe to
gate termination on.

`engine.validate_script` / `validate_workspace` is the single entry point shared
by the agent's tool, the harness stop gate S, and the benchmark's level-1
scoring. One implementation, three consumers: if the agent were gated on one rule
set and scored on another, the ablation would be uninterpretable.
"""

from adapter.validator.engine import (
    find_input_script,
    validate_script,
    validate_workspace,
)
from adapter.validator.findings import Finding, ValidationResult
from adapter.validator.rules import RuleContext, run_rules
from adapter.validator.script import Command, ParsedScript, parse_script

__all__ = [
    "Command",
    "Finding",
    "ParsedScript",
    "RuleContext",
    "ValidationResult",
    "find_input_script",
    "parse_script",
    "run_rules",
    "validate_script",
    "validate_workspace",
]
