"""Tests for M — the always-on procedural-memory primer.

M is a *measured* artifact. It is injected into every single model request and
the harness neither caps nor truncates it, so this file asserts three distinct
things: that the primer is within its budget, that it actually covers the topics
the brief requires, and that it cannot break prompt assembly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from config.loader import load_settings

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def primer() -> str:
    return load_settings(repo_root=REPO_ROOT).memory.text()


@pytest.fixture(scope="module")
def budget() -> int:
    return load_settings(repo_root=REPO_ROOT).memory.max_chars


# --------------------------------------------------------------------------- #
# budget
# --------------------------------------------------------------------------- #


def test_primer_is_within_budget(primer: str, budget: int) -> None:
    """The hard guard. Grows here mean every request pays more, forever."""
    assert len(primer) <= budget, (
        f"primer is {len(primer)} chars, budget is {budget}. "
        "Either cut content deliberately or raise memory.max_chars in config.yaml "
        "with the new measurement recorded there."
    )


def test_primer_is_not_suspiciously_small(primer: str, budget: int) -> None:
    """A budget far above the real size has stopped guarding anything.

    This catches the failure mode where someone raises ``max_chars`` to unblock a
    test and then never revisits it.
    """
    assert len(primer) > budget * 0.5, (
        f"primer is {len(primer)} chars against a {budget} budget: the budget has "
        "drifted far above the content and no longer guards anything."
    )


def test_primer_token_cost_is_reported(primer: str) -> None:
    """Record the cost explicitly so a change is visible in test output.

    Uses the harness's own estimate (`CHARS_PER_TOKEN = 4`).
    """
    estimated_tokens = len(primer) // 4
    print(f"\nM primer: {len(primer)} chars, ~{estimated_tokens} tokens per request")
    assert estimated_tokens > 0


# --------------------------------------------------------------------------- #
# content coverage — the topics the brief requires M to carry
# --------------------------------------------------------------------------- #

REQUIRED_TOPICS: dict[str, tuple[str, ...]] = {
    "command ordering": ("order", "before"),
    "units conventions": ("units", "metal", "real"),
    "atom_style": ("atom_style",),
    "system creation": ("lattice", "region", "create_box", "create_atoms"),
    "read_data": ("read_data",),
    "force field": ("pair_style", "pair_coeff"),
    "neighbor": ("neighbor", "neigh_modify"),
    "velocity": ("velocity",),
    "integrators": ("nve", "nvt", "npt"),
    "compute": ("compute",),
    "thermo": ("thermo",),
    "timestep": ("timestep",),
    "run": ("run",),
    "mass": ("mass",),
}


@pytest.mark.parametrize(("topic", "keywords"), sorted(REQUIRED_TOPICS.items()))
def test_primer_covers_required_topic(primer: str, topic: str, keywords: tuple[str, ...]) -> None:
    lowered = primer.lower()
    for keyword in keywords:
        assert keyword.lower() in lowered, f"M must cover {topic!r}: missing {keyword!r}"


# The paper's LAMMPS port names these pitfalls explicitly; M is where they live.
REQUIRED_PITFALLS: dict[str, tuple[str, ...]] = {
    "LJ lattice density semantics": ("reduced", "number density"),
    "region ordering": ("region", "create_box"),
    "unfix before switching integrators": ("unfix",),
    "SLLOD Couette pattern": ("sllod",),
    "MSD compute syntax": ("msd", "com yes"),
}


@pytest.mark.parametrize(("pitfall", "keywords"), sorted(REQUIRED_PITFALLS.items()))
def test_primer_carries_named_pitfall(primer: str, pitfall: str, keywords: tuple[str, ...]) -> None:
    lowered = primer.lower()
    for keyword in keywords:
        assert keyword.lower() in lowered, f"M must carry pitfall {pitfall!r}: missing {keyword!r}"


BENCHMARK_TASKS: tuple[str, ...] = (
    "melting",
    "equilibration",
    "msd",
    "tension",
    "nanoindentation",
)


@pytest.mark.parametrize("task", BENCHMARK_TASKS)
def test_primer_covers_each_benchmark_task(primer: str, task: str) -> None:
    """M must say something specific about each of the five in-scope tasks."""
    assert task in primer.lower(), f"M must cover benchmark task {task!r}"


def test_primer_warns_about_the_timestep_default(primer: str) -> None:
    """`timestep` defaults to 0.0 — a silent wrong-physics failure, so it is
    called out rather than merely listed."""
    assert "0.0" in primer
    assert "default" in primer.lower()


def test_primer_points_at_retrieval_for_details(primer: str) -> None:
    """M must not pretend to be the full reference; R covers syntax."""
    assert "search_lammps" in primer


# --------------------------------------------------------------------------- #
# assembly safety
# --------------------------------------------------------------------------- #


#: The one variable the plugin registers and interpolates. Everything else must
#: be literal text, because an unregistered reference fails prompt assembly.
REGISTERED_VARIABLES = frozenset({"siga_task"})


def test_primer_references_only_registered_variables(primer: str) -> None:
    """A `{{name}}` the plugin has not registered makes prompt assembly FAIL.

    `renderPrompt` interpolates strictly: an unresolved reference is an assembly
    error, not an empty string. So a stray brace pair can take the whole agent
    down. This asserts the primer references exactly the variables
    `harness/siga-plugin/src/index.ts` registers.
    """
    references = set(re.findall(r"\{\{\s*([^}]*?)\s*\}\}", primer))
    unregistered = references - REGISTERED_VARIABLES
    assert not unregistered, (
        f"primer references variables the plugin does not register: {sorted(unregistered)}. "
        f"Registered: {sorted(REGISTERED_VARIABLES)}"
    )
    # And the registration must not go stale: a registered-but-unused variable is
    # dead weight in the plugin.
    assert references == REGISTERED_VARIABLES, (
        f"primer references {sorted(references)} but the plugin registers "
        f"{sorted(REGISTERED_VARIABLES)}; keep them in step."
    )


def test_primer_binds_the_active_task(primer: str) -> None:
    """M is task-aware: the benchmark driver's task id reaches the primer."""
    assert "{{siga_task}}" in primer


def test_primer_does_not_contain_an_unclosed_brace_pair(primer: str) -> None:
    """A lone `{{` with no later `}}` is treated as literal prose by design."""
    for index, line in enumerate(primer.splitlines(), start=1):
        if "{{" in line:
            assert "}}" in line, f"line {index}: `{{{{` without a closing `}}}}`: {line!r}"


def test_primer_starts_with_a_heading(primer: str) -> None:
    assert primer.lstrip().startswith("# ")


def test_primer_sections_are_well_formed(primer: str) -> None:
    """Every `##` section must have a non-empty body."""
    sections = re.split(r"^## ", primer, flags=re.MULTILINE)[1:]
    assert sections, "primer has no `##` sections"
    for section in sections:
        title, _, body = section.partition("\n")
        assert title.strip(), "empty section title"
        assert body.strip(), f"section {title!r} has no body"


def test_primer_is_pure_ascii_or_documented(primer: str) -> None:
    """Non-ASCII is allowed (Greek letters carry real meaning for LJ), but the
    file must at least be valid UTF-8 with no replacement characters, which would
    signal an encoding accident."""
    assert "\ufffd" not in primer
