"""OVITO renders in a child process, because it is not thread-safe.

Driving OVITO from a worker thread did not raise — it **segfaulted**. The crash
report named the frame:

    ovito_bindings.so  Ovito::PythonLongRunningOperation::PythonLongRunningOperation(bool)
    ovito_bindings.so  ...defineIOBindings...  FileImporter ... (QUrl const&)
    Python             thread_run -> context_run -> partial_call

An `asyncio.to_thread(render_structure, ...)` call site produces exactly that
stack. The console process died with it, and the launcher then shut the harness
down as well, so the symptom a user reported was "the harness keeps dying" —
three steps from the click that caused it.

Two properties are pinned here: that a crash is *reported* as a crash rather than
as a failed render, and that the observables actually compute. The second was
broken separately and silently: `if table.xy():` raised "truth value of an array
with more than one element is ambiguous", the bare `except` turned that into
"observables failed", and every render was missing its RDF numbers.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from adapter.visualize import VisualisationError, render_structure
from start import _exit_description

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A four-atom fcc cell, which is enough for OVITO to read and analyse.
MINIMAL_DUMP = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
4
ITEM: BOX BOUNDS pp pp pp
0.0 3.6
0.0 3.6
0.0 3.6
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
2 1 0.0 1.8 1.8
3 1 1.8 0.0 1.8
4 1 1.8 1.8 0.0
"""


def _ovito_available() -> bool:
    try:
        import ovito  # noqa: F401
    except Exception:
        return False
    return True


requires_ovito = pytest.mark.skipif(
    not _ovito_available(), reason="OVITO is not installed in this environment"
)


# --------------------------------------------------------------------------- #
# a signal death must be distinguishable from a described failure
# --------------------------------------------------------------------------- #


def test_a_segfault_really_does_report_a_negative_returncode() -> None:
    """The crash handling rests on this: Python reports a signal as -N.

    Asserted by causing a real one, so the assumption cannot rot silently and
    leave the crash path matching nothing.
    """
    completed = subprocess.run(
        [sys.executable, "-c", "import ctypes; ctypes.string_at(0)"],
        capture_output=True,
    )
    assert completed.returncode < 0, "a segfault should surface as a negative return code"
    assert -completed.returncode in (10, 11), "expected SIGBUS or SIGSEGV"


def test_render_in_subprocess_names_the_signal_instead_of_blaming_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash costs one image; it does not mean the simulation failed.

    Saying "渲染失败" for a segfault sends the reader to inspect a script that is
    fine, so the message has to separate the two and say what is still intact.
    """
    from web.backend import app as backend

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=-11, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    source = tmp_path / "traj.dump"
    source.write_text(MINIMAL_DUMP)

    with pytest.raises(VisualisationError) as caught:
        backend._render_in_subprocess(source, tmp_path / "out.png", None)

    message = str(caught.value)
    assert "信号 11" in message
    assert "崩溃" in message
    # It must not imply the run itself was bad.
    assert "脚本" in message and "完好" in message


def test_render_in_subprocess_reports_a_clean_failure_with_its_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A described failure and a crash are different events with different remedies."""
    from web.backend import app as backend

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="ValueError: bad dump header\n"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    source = tmp_path / "traj.dump"
    source.write_text(MINIMAL_DUMP)

    with pytest.raises(VisualisationError) as caught:
        backend._render_in_subprocess(source, tmp_path / "out.png", None)
    message = str(caught.value)
    assert "退出码 1" in message
    assert "bad dump header" in message


def test_a_timed_out_render_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from web.backend import app as backend

    def fake_run(*_args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="visualize_cli", timeout=kwargs.get("timeout", 1))

    monkeypatch.setattr(subprocess, "run", fake_run)
    source = tmp_path / "traj.dump"
    source.write_text(MINIMAL_DUMP)

    with pytest.raises(VisualisationError) as caught:
        backend._render_in_subprocess(source, tmp_path / "out.png", None)
    assert "超" in str(caught.value)


def test_exit_description_translates_a_negative_code_into_a_signal_name() -> None:
    assert _exit_description(-11) == "被信号终止：SIGSEGV（段错误）"
    assert _exit_description(-9) == "被信号终止：SIGKILL"
    assert _exit_description(0) == "退出码 0"
    assert _exit_description(None) == "状态未知"


# --------------------------------------------------------------------------- #
# the observables must actually be computed
# --------------------------------------------------------------------------- #


@requires_ovito
def test_observables_include_the_rdf_peak(tmp_path: Path) -> None:
    """The RDF numbers were missing from every render because of a NumPy misuse.

    `table.xy()` returns an array; `if array:` raises, the surrounding `except`
    converted that into a warning, and the result was a picture with no
    observables and a reason nobody could act on.
    """
    source = tmp_path / "cell.dump"
    source.write_text(MINIMAL_DUMP)
    output = tmp_path / "cell.png"

    result = render_structure(source, output)

    assert result.atoms == 4
    assert result.warning is None, f"observables reported a problem: {result.warning}"
    assert "rdf_first_peak_r" in result.observables
    assert result.observables["rdf_first_peak_r"] > 0
    assert result.observables["rdf_points"] > 0


@requires_ovito
def test_a_render_produces_an_image_with_content_not_a_blank_canvas(tmp_path: Path) -> None:
    """`render_image` succeeds on an empty scene, so a tiny valid PNG proves nothing."""
    source = tmp_path / "cell.dump"
    source.write_text(MINIMAL_DUMP)
    output = tmp_path / "cell.png"

    render_structure(source, output)

    assert output.is_file()
    # The module already refuses implausibly small output; this asserts the file
    # is comfortably past that line rather than exactly on it.
    assert output.stat().st_size > 5000


@requires_ovito
def test_the_cli_exits_zero_on_success_and_writes_its_result(tmp_path: Path) -> None:
    """The console reads this contract, so it is asserted against the real CLI."""
    source = tmp_path / "cell.dump"
    source.write_text(MINIMAL_DUMP)
    output = tmp_path / "cell.png"
    result_path = tmp_path / "result.json"

    completed = subprocess.run(
        [sys.executable, "-m", "adapter.visualize_cli", str(source), str(output),
         "--result", str(result_path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert result_path.is_file()
    import json

    payload = json.loads(result_path.read_text())
    assert payload["ok"] is True
    assert payload["atoms"] == 4


@requires_ovito
def test_the_cli_reports_a_bad_file_as_a_clean_failure(tmp_path: Path) -> None:
    """Exit 1 with a JSON reason, so the console can show it without guessing."""
    missing = tmp_path / "absent.dump"
    output = tmp_path / "absent.png"
    result_path = tmp_path / "result.json"

    completed = subprocess.run(
        [sys.executable, "-m", "adapter.visualize_cli", str(missing), str(output),
         "--result", str(result_path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )

    assert completed.returncode == 1
    import json

    payload = json.loads(result_path.read_text())
    assert payload["ok"] is False
    assert payload["error"]


# --------------------------------------------------------------------------- #
# the file panel must count what the run produced
# --------------------------------------------------------------------------- #


def test_the_file_list_excludes_the_render_cache_and_the_metadata(tmp_path: Path) -> None:
    """A run that wrote five outputs reported nine files.

    The metadata this console writes and the PNGs cached from rendering both live
    in the workspace, and both are real files — but neither is something the
    simulation produced. The count is the point of the panel, so counting them
    overstates what the run did.
    """
    from web.backend.app import _workspace_files

    (tmp_path / "in.melt").write_text("units lj\n")
    (tmp_path / "log.lammps").write_text("Step Temp\n")
    (tmp_path / "solid.dump").write_text("ITEM: TIMESTEP\n")
    (tmp_path / "run.json").write_text("{}")
    cache = tmp_path / ".siga-visual"
    cache.mkdir()
    (cache / "solid.dump.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    names = [f["name"] for f in _workspace_files(tmp_path)]

    assert names == ["in.melt", "log.lammps", "solid.dump"]
    assert "run.json" not in names
    assert not any(".siga-visual" in n for n in names)


def test_the_file_list_reports_sizes(tmp_path: Path) -> None:
    from web.backend.app import _workspace_files

    (tmp_path / "traj.dump").write_bytes(b"x" * 1234)
    files = _workspace_files(tmp_path)
    assert files == [{"name": "traj.dump", "size": 1234}]
