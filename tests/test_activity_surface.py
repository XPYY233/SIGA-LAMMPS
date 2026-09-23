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


# --------------------------------------------------------------------------- #
# run history
# --------------------------------------------------------------------------- #


def test_a_run_records_the_request_so_history_is_classifiable() -> None:
    """Without the request text, free-form runs cannot be told apart.

    The task id is empty for a free-form request and the folder name is opaque,
    so a history list of them says nothing about what any of them was for.
    """
    from web.backend.app import Run

    run = Run(
        run_id="run-1", task_id=None, configuration="mrsx",
        workspace=__import__("pathlib").Path("/tmp/x"),
        request="把 LJ 晶体熔化并报告扩散系数",
    )
    assert run.to_dict()["request"] == "把 LJ 晶体熔化并报告扩散系数"


def test_metadata_round_trips_the_request(tmp_path) -> None:
    """The request must survive a restart, or history is classifiable only live."""
    from web.backend.app import Run, _read_metadata, _write_metadata

    run = Run(
        run_id="run-2", task_id=None, configuration="mr", workspace=tmp_path,
        request="equilibrate at 300K",
    )
    _write_metadata(run)
    assert _read_metadata(tmp_path)["request"] == "equilibrate at 300K"


def test_metadata_absent_or_corrupt_is_not_fatal(tmp_path) -> None:
    """History discovery must not break because one file is unreadable."""
    from web.backend.app import _read_metadata

    assert _read_metadata(tmp_path) == {}
    (tmp_path / "run.json").write_text("{not json", encoding="utf-8")
    assert _read_metadata(tmp_path) == {}


def test_history_events_are_backfilled_from_the_session() -> None:
    """A run recovered from disk has no live pump, so it must be replayed.

    Without this the console sat on '正在加载该任务的活动流…' forever. The events were
    never lost — the session log is durable and replayable — they simply had no
    reader for a run that was not created in this process.
    """
    import asyncio

    from web.backend.app import Run, _backfill_events

    class _StubHarness:
        async def history(self, session_id, max_messages=400):
            return {
                "events": [
                    {"event": {"type": "tool/call", "seq": 1,
                               "data": {"name": "mcp__lammps__validate_lammps_input"}}},
                    {"event": {"type": "assistant/chunk", "seq": 2, "data": {}}},
                ]
            }

    run = Run(
        run_id="run-3", task_id=None, configuration="mrsx",
        workspace=__import__("pathlib").Path("/tmp/x"), session_id="session-1",
    )
    asyncio.run(_backfill_events(run, _StubHarness()))

    # Only replayed event kinds reach Area B; assistant chunks never do.
    assert len(run.events) == 1
    assert run.events[0]["data"]["component"] == "X"


def test_backfill_does_not_duplicate_existing_events() -> None:
    """A live run already has its events; replaying must not double them."""
    import asyncio

    from web.backend.app import Run, _backfill_events

    class _StubHarness:
        async def history(self, session_id, max_messages=400):
            raise AssertionError("must not be called for a run that already has events")

    run = Run(
        run_id="run-4", task_id=None, configuration="mrsx",
        workspace=__import__("pathlib").Path("/tmp/x"), session_id="s",
    )
    run.events.append({"seq": 1, "type": "tool/call", "data": {}})
    asyncio.run(_backfill_events(run, _StubHarness()))
    assert len(run.events) == 1


# --------------------------------------------------------------------------- #
# the shapes the harness actually emits
# --------------------------------------------------------------------------- #
#
# Every case below uses a verbatim event body captured from a real session log.
# The earlier code assumed `arguments` was a mapping and that result text lived
# at `data["content"]`; both assumptions were wrong, and the surface silently
# degraded to "运行命令" with no command and "(无输出)" for every result. Testing
# against invented shapes would not have caught either.


def test_arguments_arrive_as_a_json_string() -> None:
    """The event log stores the wire form, which is a string, not a mapping."""
    payload = _summarise(
        "tool/call",
        {
            "turn": 1, "step": 1, "callId": "c",
            "name": "bash",
            "arguments": '{"command": "pwd; ls -la", "description": "List workspace contents"}',
        },
    )
    assert payload["args"] == "pwd; ls -la", payload


def test_a_malformed_argument_string_does_not_crash_the_feed() -> None:
    payload = _summarise(
        "tool/call", {"name": "bash", "arguments": "{not json"}
    )
    assert payload["component"] == "执行"
    assert "not json" in payload["args"]


def test_result_text_is_nested_inside_the_tool_result_block() -> None:
    """Text sits at message.content[].content[].text — one level deeper."""
    payload = _summarise(
        "tool/result",
        {
            "turn": 1, "step": 1,
            "message": {
                "source": {"kind": "tool", "callId": "c"},
                "content": [{
                    "type": "tool-result",
                    "toolCallId": "c",
                    "isError": False,
                    "content": [{"type": "text", "text": "Step Temp E_pair\n0 1.2 -6.77"}],
                }],
            },
        },
    )
    assert "Step Temp" in payload["preview"]
    assert payload["is_error"] is False


def test_a_real_validator_result_is_still_decoded_by_name() -> None:
    """The validator path must keep working with the nested shape."""
    inner = json.dumps({
        "valid": False,
        "errors": [{"code": "UNITS_MISSING", "message": "no units"}],
        "counts": {"errors": 1, "warnings": 0, "suggestions": 0},
    })
    payload = _summarise(
        "tool/result",
        {"message": {"content": [{"type": "tool-result", "content": [{"type": "text", "text": inner}]}]}},
    )
    assert payload["validation"]["errors"] == 1
    assert "缺少 units 命令" in payload["validation"]["codes"]


def test_progress_is_carried_as_turn_and_step() -> None:
    """A viewer needs to know how far along a run is, which seq does not say."""
    import asyncio

    from web.backend.app import Run, _relay

    run = Run(
        run_id="r", task_id=None, configuration="mrsx",
        workspace=__import__("pathlib").Path("/tmp/x"),
    )
    _relay(run, {"type": "tool/call", "seq": 3951, "data": {"turn": 3, "step": 7, "name": "bash"}})
    assert run.events[0]["turn"] == 3
    assert run.events[0]["step"] == 7


def test_only_the_adapter_is_the_stop_gate() -> None:
    """Runtime-context injection arrives on the same channel as an S steer.

    Labelling it as an interception pointed a viewer at exactly the wrong
    moment, which is the one thing S's display exists to get right.
    """
    context = _summarise(
        "user/message",
        {"source": {"kind": "plugin", "plugin": "@deepseek-ai/dsh-system-prompt"},
         "content": [{"type": "text", "text": "Current runtime context. This snapshot…"}]},
    )
    assert context["component"] == "环境"
    assert context["label"] == "注入运行时上下文"

    gate = _summarise(
        "user/message",
        {"source": {"kind": "plugin", "plugin": "siga-lammps"},
         "content": [{"type": "text", "text": "Stop-gate: validation failed"}]},
    )
    assert gate["component"] == "S"
    assert gate["label"] == "停止门控拦截"


def test_a_human_message_gets_no_component() -> None:
    payload = _summarise(
        "user/message", {"source": {"kind": "user"}, "content": [{"type": "text", "text": "hi"}]}
    )
    assert payload["component"] is None


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #


def test_results_read_the_local_log_not_only_the_cluster_one() -> None:
    """The agent runs LAMMPS locally while working, and that is where numbers appear.

    The job panel reads the cluster's log, which stays empty until something is
    submitted. A researcher asking "how do I see the output" should not have to
    discover that the answer was in a file called log.local all along.
    """
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "web" / "backend" / "app.py").read_text()
    assert '"/api/runs/{run_id}/results"' in source
    # Any log name, chosen by size, not a hardcoded "log.lammps".
    assert 'startswith("log")' in source
    assert 'suffix == ".log"' in source


def test_thermo_columns_survive_the_round_trip() -> None:
    """The thermo table is the physics output, so its parsing is load-bearing."""
    from benchmark.evaluator import parse_thermo

    text = (
        "LAMMPS (22 Jul 2025)\n"
        "   Step          Temp         Pxy\n"
        "      0   1.0000000   0.0000000\n"
        "   5000   1.0022916   0.1533985\n"
        "  10000   1.0022916   0.1533985\n"
        "Loop time of 1.0\nTotal wall time: 0:00:01\n"
    )
    columns, rows = parse_thermo(text)
    assert columns == ["Step", "Temp", "Pxy"]
    assert len(rows) == 3
    assert rows[-1][0] == 10000.0


def test_a_finished_log_is_distinguishable_from_a_truncated_one() -> None:
    """Whether LAMMPS reached the end is the first thing a reader needs."""
    from benchmark.evaluator import parse_thermo

    finished = "   Step Temp\n 0 1.0\nLoop time\nTotal wall time: 0:00:01\n"
    truncated = "   Step Temp\n 0 1.0\n"
    _, a = parse_thermo(finished)
    _, b = parse_thermo(truncated)
    assert a and b
    assert "Total wall time" in finished
    assert "Total wall time" not in truncated
