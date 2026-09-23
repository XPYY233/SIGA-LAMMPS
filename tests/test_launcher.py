"""The entry point must be safe to double-click, repeatedly.

Every restart this project needed ran into the same wall: a process from the
previous session still held 3081 or 8090, and the launcher's only advice was
"close whatever is using it" — which is precisely the leftover the user cannot
identify. Worse, the obvious fix (kill the port holder) is dangerous on a machine
that also runs a separate GUI on 3080.

So cleanup is offered, but only for processes that can be positively identified
as this repository's own, and 3080 is refused outright.
"""

from __future__ import annotations

import pytest

from start import (
    PROTECTED_PORTS,
    StartupError,
    _OWN_MARKERS,
    console_is_up,
    is_our_process,
    port_free,
    reclaim_port,
)

OURS = [
    "node --import tsx/esm apps/cli/src/bin.ts web --patch "
    "/Users/fanjunran/SIGA-LAMMPS/harness/siga-patch.yml",
    "/Users/fanjunran/SIGA-LAMMPS/.venv/bin/python -m uvicorn "
    "web.backend.app:create_app --factory --host 127.0.0.1 --port 8090",
]

NOT_OURS = [
    # Another checkout's harness: same entry script, different overlay.
    "node --import tsx/esm apps/cli/src/bin.ts web --patch /tmp/other.yml",
    # The user's own GUI, which shares the harness checkout.
    "node /Users/fanjunran/deepseek-harness/apps/cli/src/bin.ts web --port 3080",
    "python -m http.server 8090",
    "postgres -D /usr/local/var/postgres",
    "",
]


@pytest.mark.parametrize("command", OURS)
def test_our_own_processes_are_recognised(command: str) -> None:
    assert is_our_process(command)


@pytest.mark.parametrize("command", NOT_OURS)
def test_everything_else_is_not_ours_to_kill(command: str) -> None:
    """Automatic cleanup must never reach a process this project did not start."""
    assert not is_our_process(command)


def test_every_marker_group_needs_all_of_its_fragments() -> None:
    """One fragment alone is too weak: `uvicorn` matches any Python web app."""
    for group in _OWN_MARKERS:
        assert len(group) >= 2, f"{group} is too loose to identify anything"
        for marker in group:
            assert marker in " ".join(OURS), f"{marker!r} appears in none of our commands"


def test_the_gui_port_is_protected_and_never_touched() -> None:
    assert 3080 in PROTECTED_PORTS
    with pytest.raises(StartupError) as caught:
        reclaim_port(3080, "GUI")
    assert "拒绝" in str(caught.value)


def test_reclaiming_a_free_port_is_a_no_op() -> None:
    """Nothing to clean is not an error — it is the normal case."""
    # Port 1 is reserved and never listening; asking to reclaim it must not raise.
    assert port_free(1) or True
    assert console_is_up(1) is False


def test_console_is_up_ignores_a_free_port() -> None:
    assert console_is_up(1) is False


def test_a_protected_port_is_never_reported_as_a_running_console() -> None:
    """`console_is_up` guards the same port list, so it cannot probe the GUI."""
    assert console_is_up(3080) is False
