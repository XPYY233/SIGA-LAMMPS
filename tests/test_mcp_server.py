"""Tests for the MCP tool surface.

Verified over a real stdio handshake rather than by inspecting the server object,
because the handshake is what the harness does. The registered names matter
exactly: the model sees `mcp__<server>__<tool>`, and the primer and the stop gate
both name the qualified form, so a rename here would silently make their
instructions wrong.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

EXPECTED_TOOLS = {"search_lammps", "validate_lammps_input"}

pytestmark = pytest.mark.skipif(
    not VENV_PYTHON.is_file(), reason="adapter virtualenv not present"
)


async def _call(tool: str, args: dict) -> tuple[bool, dict]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=str(VENV_PYTHON), args=["-m", "adapter.mcp_server"], cwd=str(REPO_ROOT)
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = {t.name for t in listed.tools}
            assert EXPECTED_TOOLS <= names, f"missing tools: {EXPECTED_TOOLS - names}"
            result = await session.call_tool(tool, args)
            text = result.content[0].text if result.content else "{}"
            return result.is_error, json.loads(text)


def call(tool: str, args: dict) -> tuple[bool, dict]:
    return asyncio.run(_call(tool, args))


def test_both_tools_are_exposed() -> None:
    """A handshake that lists the tools is itself the assertion."""
    is_error, _ = call("validate_lammps_input", {"workspace": "."})
    assert is_error is False


def test_validate_returns_the_agreed_contract(tmp_path: Path) -> None:
    (tmp_path / "in.test").write_text(
        "units lj\natom_style atomic\nlattice fcc 0.8442\n"
        "region b block 0 4 0 4 0 4\ncreate_box 1 b\ncreate_atoms 1 b\n"
        "mass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0 2.5\n"
        "fix 1 all nve\ntimestep 0.005\nrun 100\n",
        encoding="utf-8",
    )
    is_error, payload = call("validate_lammps_input", {"workspace": str(tmp_path)})
    assert is_error is False
    assert set(payload) >= {"valid", "errors", "warnings", "suggestions", "counts"}
    assert payload["valid"] is True


def test_validate_reports_errors_for_a_broken_script(tmp_path: Path) -> None:
    (tmp_path / "in.test").write_text("atom_style atomic\nrun 100\n", encoding="utf-8")
    _, payload = call("validate_lammps_input", {"workspace": str(tmp_path)})
    assert payload["valid"] is False
    assert payload["counts"]["errors"] > 0
    codes = {f["code"] for f in payload["errors"]}
    assert "UNITS_MISSING" in codes


def test_validate_returns_a_structured_error_when_there_is_nothing_to_check(
    tmp_path: Path,
) -> None:
    """An empty workspace is a run condition, not an invalid script.

    Returning `valid: false` here would report a defect that does not exist and
    would make the stop gate block a turn for a workspace it cannot assess.
    """
    _, payload = call("validate_lammps_input", {"workspace": str(tmp_path)})
    assert payload.get("error") == "no_input_script"
    assert payload["valid"] is None


def test_validate_unknown_task_is_a_structured_error(tmp_path: Path) -> None:
    (tmp_path / "in.test").write_text("units lj\nrun 0\n", encoding="utf-8")
    _, payload = call("validate_lammps_input", {"workspace": str(tmp_path), "task": "nope"})
    assert payload.get("error") == "unknown_task"


def test_validate_can_check_against_a_named_task(tmp_path: Path) -> None:
    (tmp_path / "in.test").write_text("units lj\natom_style atomic\nrun 0\n", encoding="utf-8")
    _, payload = call("validate_lammps_input", {"workspace": str(tmp_path), "task": "lj_melt"})
    assert payload["task"] == "lj_melt"
    codes = {f["code"] for f in payload["errors"]}
    assert "TASK_REQUIRED_COMMAND_MISSING" in codes


def test_search_returns_sourced_passages() -> None:
    _, payload = call("search_lammps", {"query": "fix deform erate", "k": 3})
    assert payload.get("count", 0) > 0, payload
    for hit in payload["results"]:
        # A hit without a citable source is not checkable by the agent.
        assert hit["source"]
        assert hit["snippet"].strip()


def test_search_reports_the_backend() -> None:
    """A retrieval result is not comparable across backends, so it is declared."""
    _, payload = call("search_lammps", {"query": "lennard jones cutoff"})
    assert payload["backend"] in {"bm25"}
