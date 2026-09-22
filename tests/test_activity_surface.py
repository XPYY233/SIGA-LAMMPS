"""Tests for what Area B is allowed to show, and how it explains itself.

The requirement is that a researcher watching a run can tell real work from
apparent idling: which adapter component acted, what it was asked to do, and
what came back. "Called bash()" satisfies none of those.
"""

from __future__ import annotations

import json

from web.backend.app import _argument_summary, _role_of, _summarise


def test_retrieval_and_validation_are_attributed_to_their_components() -> None:
    """R and X are the adapter's tool interface, and must be labelled as such."""
    component, label, why = _role_of("mcp__lammps__search_lammps")
    assert component == "R"
    assert why.strip()

    component, label, why = _role_of("mcp__lammps__validate_lammps_input")
    assert component == "X"
    assert why.strip()


def test_hpc_tools_are_attributed() -> None:
    for name in ("hpc_submit_job", "hpc_job_status", "hpc_cancel_job"):
        assert _role_of(name)[0] == "HPC"


def test_generic_tools_still_get_a_role() -> None:
    """An unlisted tool must not render as a bare name with no explanation."""
    component, label, why = _role_of("some_future_tool")
    assert component and label and why


def test_arguments_are_summarised_not_dropped() -> None:
    """The arguments are the difference between work and spinning."""
    assert "ls" in _argument_summary("bash", {"command": "ls -la"})
    assert "in.melt" in _argument_summary("write", {"file_path": "/w/in.melt"})
    assert _argument_summary("bash", {}) == ""


def test_a_long_command_is_truncated() -> None:
    summary = _argument_summary("bash", {"command": "x" * 500})
    assert len(summary) <= 221


def test_a_tool_call_carries_component_purpose_and_arguments() -> None:
    payload = _summarise(
        "tool/call",
        {"name": "mcp__lammps__validate_lammps_input", "callId": "c1",
         "arguments": {"workspace": "/w/run-1"}},
    )
    assert payload["component"] == "X"
    assert payload["label"]
    assert payload["why"]
    assert "run-1" in payload["args"]


def test_a_stop_gate_steer_is_attributed_to_s() -> None:
    """S is the one component whose action is invisible without labelling.

    It appears only as a plugin-sourced user message, which without attribution
    looks like the researcher talking to themselves.
    """
    payload = _summarise(
        "user/message",
        {"source": {"kind": "plugin", "plugin": "siga-lammps"},
         "content": [{"type": "text", "text": "Stop-gate: validation failed"}]},
    )
    assert payload["component"] == "S"
    assert payload["label"] == "停止门控拦截"
    assert "拒绝结束" in payload["why"]


def test_a_human_message_is_not_attributed_to_a_component() -> None:
    payload = _summarise(
        "user/message", {"source": {"kind": "user"}, "content": [{"type": "text", "text": "hello"}]}
    )
    assert payload["component"] is None


def test_validation_results_are_summarised_by_finding_name() -> None:
    """A validator verdict is the most informative event, so it is decoded.

    Raw JSON would push the reader to interpret codes; naming them turns the
    event into "X found an ordering error and a missing timestep".
    """
    inner = json.dumps({
        "valid": False,
        "errors": [{"code": "ORDER_VIOLATION", "message": "..."},
                   {"code": "TIMESTEP_MISSING", "message": "..."}],
        "counts": {"errors": 2, "warnings": 0, "suggestions": 0},
    })
    payload = _summarise("tool/result", {"callId": "c1", "content": [{"type": "text", "text": inner}]})
    validation = payload["validation"]
    assert validation["valid"] is False
    assert validation["errors"] == 2
    assert "命令顺序错误" in validation["codes"]
    assert "缺少 timestep（默认 0.0，动力学无意义）" in validation["codes"]


def test_a_non_validation_result_is_left_alone() -> None:
    payload = _summarise("tool/result", {"callId": "c", "content": [{"type": "text", "text": "ok"}]})
    assert "validation" not in payload
    assert payload["preview"] == "ok"


def test_reasoning_is_never_forwarded() -> None:
    """The brief forbids exposing hidden reasoning; only durable action is shown."""
    payload = _summarise("assistant/message", {"message": {"content": [{"type": "text", "text": "secret"}]}})
    assert "secret" not in json.dumps(payload)
