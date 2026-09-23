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
    # `neighbor` belongs in M's ordering skeleton; `neigh_modify` delay/check
    # tuning does not — it is retrievable detail, and adding it would push a
    # task-independent but low-frequency knob into always-on context.
    "neighbor": ("neighbor",),
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


# The paper's LAMMPS port names these pitfalls explicitly, and they are the kind
# of high-frequency, task-independent mistake that belongs always-on.
REQUIRED_PITFALLS: dict[str, tuple[str, ...]] = {
    "LJ lattice density semantics": ("reduced", "number density"),
    "region ordering": ("region", "create_box"),
    "unfix before switching integrators": ("unfix",),
    "timestep default": ("0.0",),
}


@pytest.mark.parametrize(("pitfall", "keywords"), sorted(REQUIRED_PITFALLS.items()))
def test_primer_carries_named_pitfall(primer: str, pitfall: str, keywords: tuple[str, ...]) -> None:
    lowered = primer.lower()
    for keyword in keywords:
        assert keyword.lower() in lowered, f"M must carry pitfall {pitfall!r}: missing {keyword!r}"


def test_primer_gives_tool_usage_guidance(primer: str) -> None:
    """M must tell the agent *when* to reach for R and X, not just exist.

    Without this, an agent that does not know it should search is no better off
    than one with no retrieval layer at all.
    """
    lowered = primer.lower()
    assert "mcp__lammps__search_lammps" in primer
    assert "mcp__lammps__validate_lammps_input" in primer
    # Guidance, not merely a mention: it must say when to call them.
    assert "before" in lowered


def test_primer_ends_with_a_completion_checklist(primer: str) -> None:
    """A checkable finishing list, so "done" has a definition the agent can use.

    This is advisory and complements S; it does not replace the enforced gate.
    """
    lowered = primer.lower()
    assert "before you finish" in lowered or "checklist" in lowered
    assert primer.count("- [ ]") >= 4, "the checklist needs actual checkable items"


# --------------------------------------------------------------------------- #
# the M/R boundary
# --------------------------------------------------------------------------- #

#: Content that belongs in R, not M. The project rule (docs/design-principles.md
#: §2): knowledge that is long, task-specific, or reliably retrievable is
#: retrieved on demand. M is for high-frequency procedural knowledge only.
#:
#: Note the boundary is about *recipes*, not *pitfalls*. "compute msd needs
#: com yes" is a mistake and stays in M; a step-by-step MSD walkthrough is a
#: recipe and belongs in R. This test exists because the cheapest way to break
#: the boundary is to append "just one more useful thing" to an always-on file.
MUST_NOT_APPEAR_IN_M: dict[str, str] = {
    "nanoindentation": "a per-task recipe; retrieve fix indent via R",
    "uniaxial tension": "a per-task recipe; retrieve fix deform via R",
    "melting temperature": "a per-task value; belongs in the task specification",
    "# example": "a worked example block; that is what R returns",
}


@pytest.mark.parametrize(("needle", "reason"), sorted(MUST_NOT_APPEAR_IN_M.items()))
def test_task_specific_knowledge_stays_out_of_m(primer: str, needle: str, reason: str) -> None:
    assert needle not in primer.lower(), (
        f"M contains {needle!r}, which is out of scope: {reason}. "
        "Always-on context is billed on every request, and task-specific detail "
        "belongs in R where it is fetched only when needed."
    )


def test_m_stays_within_its_token_ceiling(primer: str) -> None:
    """The M/R boundary, expressed as the project's token ceiling.

    M is allowed up to 2000 tokens. At the harness's CHARS_PER_TOKEN = 4 that is
    8000 characters. A breach means knowledge is being kept always-on that
    should be retrieved on demand.
    """
    tokens = len(primer) / 4
    assert len(primer) <= 8000, (
        f"M is {len(primer)} chars (~{tokens:.0f} tokens), over the 2000-token "
        "ceiling. Move detail to R rather than growing M."
    )


def test_primer_warns_about_the_timestep_default(primer: str) -> None:
    """`timestep` defaults to 0.0 — a silent wrong-physics failure, so it is
    called out rather than merely listed."""
    assert "0.0" in primer
    assert "default" in primer.lower()


def test_primer_points_at_retrieval_for_details(primer: str) -> None:
    """M must not pretend to be the full reference; R covers syntax."""
    assert "mcp__lammps__search_lammps" in primer


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
