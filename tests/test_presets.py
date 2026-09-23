"""Tests for the ablation configurations.

The controlled-comparison requirement is that the adapter configuration is the
ONLY variable. These tests assert the composition of each configuration rather
than running it, so a mistake is caught here rather than after an expensive
benchmark sweep.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "harness"))

import make_patch  # noqa: E402

CONFIGURATIONS = tuple(make_patch.PRESET_SUMMARIES)


def test_the_four_configurations_are_declared() -> None:
    assert CONFIGURATIONS == ("vanilla", "m", "mr", "mrsx")


def test_vanilla_mounts_nothing() -> None:
    """The capability floor must be genuinely empty.

    If vanilla mounted anything the baseline would be contaminated, and every
    measured improvement would be relative to the wrong thing.
    """
    assert make_patch.component_rows("vanilla", None, None).strip() == ""


@pytest.mark.parametrize("preset", ["m", "mr", "mrsx"])
def test_m_is_present_in_every_grounded_configuration(preset: str) -> None:
    rows = make_patch.component_rows(preset, None, None)
    assert "siga-memory" in rows


@pytest.mark.parametrize(
    ("preset", "expect_mcp"),
    [("m", False), ("mr", True), ("mrsx", True)],
)
def test_retrieval_and_validation_arrive_only_with_mcp(preset: str, expect_mcp: bool) -> None:
    """R and X are the same server, so they cannot be separated by this design.

    Stated as a test rather than a comment: the four configurations are
    vanilla, M, M+R, and M+R+X+S, not every combination of four binary factors.
    Separating R from X would need a second MCP server.
    """
    rows = make_patch.component_rows(preset, None, None)
    assert ("siga-mcp" in rows) is expect_mcp


@pytest.mark.parametrize(
    ("preset", "expect_gate"),
    [("m", False), ("mr", False), ("mrsx", True)],
)
def test_the_stop_gate_is_enabled_only_in_the_full_configuration(
    preset: str, expect_gate: bool
) -> None:
    """S is a config flag on the M plugin, not a separate plugin.

    Keeping it a flag means M cannot drift between the configurations that
    include it: there is one memory implementation and one place it is defined.
    """
    rows = make_patch.component_rows(preset, None, None)
    assert f"stopGate: {'true' if expect_gate else 'false'}" in rows


def test_every_configuration_gets_the_same_task_binding() -> None:
    """Task text is frozen; a configuration-specific binding would confound.

    Only the BINDING is compared, not the whole composition: the compositions
    must differ, or there would be no experiment.
    """
    bindings = set()
    for preset in ("m", "mr", "mrsx"):
        rows = make_patch.component_rows(preset, "lj_melt", "lj_melt")
        active = [l.strip() for l in rows.splitlines() if l.strip().startswith("activeTask:")]
        gate_task = [l.strip() for l in rows.splitlines() if l.strip().startswith("taskId:")]
        bindings.add((tuple(active), tuple(gate_task)))
    assert len(bindings) == 1, f"grounded configurations bind tasks differently: {bindings}"
    active, gate_task = bindings.pop()
    assert active == ("activeTask: 'lj_melt'",)
    assert gate_task == ("taskId: 'lj_melt'",), (
        "the stop gate must validate against the same task the agent was given, "
        "or it would gate on requirements the agent was never told about"
    )


def test_the_overlay_mounts_components_rather_than_relying_on_presets() -> None:
    """A recorded decision, not an oversight.

    Agent presets are the intended per-session mechanism and they compose
    correctly, but their wiring does not reach a headless session: with the
    roster inserted and a preset selected, every session still mounted nothing,
    with no warning. Rather than depend on unverified wiring, the driver emits
    one overlay per configuration. The controlled comparison is preserved —
    same model, harness, task and settings; only the overlay differs.
    """
    overlay = make_patch.render(default_preset="mrsx")
    assert "siga-memory" in overlay
    assert "siga-mcp" in overlay
