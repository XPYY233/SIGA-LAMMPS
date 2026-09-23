"""Render one structure file in a dedicated process, and report the result.

OVITO is built on Qt, and its file importer and renderer are not safe to drive
from a worker thread. Calling it that way did not raise — it **segfaulted**,
inside `Ovito::defineIOBindings`' file-importer callback, on a thread started by
`asyncio.to_thread`. The stack was unambiguous:

    ovito_bindings.so  Ovito::PythonLongRunningOperation::PythonLongRunningOperation(bool)
    ovito_bindings.so  ...defineIOBindings...  FileImporter ... (QUrl const&)
    Python             _PyEval_EvalFrameDefault
    Python             thread_run -> context_run -> partial_call

The console process died with it. Because the launcher treated any child exiting
as a reason to shut everything down, the harness died too, and the visible
symptom was "the harness keeps dying" — three steps removed from the click that
caused it.

So OVITO now runs here, in a process of its own, on its own main thread. A crash
is then contained: the parent sees a non-zero exit, reports that OVITO crashed
and with which signal, and stays up to answer the next request.

Called as::

    python -m adapter.visualize_cli SOURCE OUTPUT [--frame N]

The result is written as JSON to the file named by --result, or to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Exit code used when OVITO reports a problem it can describe, as opposed to
#: dying. Distinguishing the two is the point of running in a child at all: one
#: is a bad file, the other is a broken environment, and the remedies differ.
EXIT_RENDER_ERROR = 1
EXIT_USAGE = 2
EXIT_CRASHED = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="visualize_cli",
        description="Render one frame of a structure file with OVITO, in isolation.",
    )
    parser.add_argument("source", help="structure or trajectory file to render")
    parser.add_argument("output", help="PNG path to write")
    parser.add_argument("--frame", type=int, default=None, help="frame index; last by default")
    parser.add_argument("--result", default="", help="write the JSON result here instead of stdout")
    args = parser.parse_args(argv)

    from adapter.visualize import VisualisationError, render_structure

    try:
        result = render_structure(
            Path(args.source),
            Path(args.output),
            frame=args.frame,
        )
    except VisualisationError as exc:
        payload = {"ok": False, "error": str(exc)}
        _emit(payload, args.result)
        return EXIT_RENDER_ERROR
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _emit(payload, args.result)
        return EXIT_RENDER_ERROR

    _emit({"ok": True, **result.to_dict()}, args.result)
    return 0


def _emit(payload: dict, destination: str) -> None:
    text = json.dumps(payload, ensure_ascii=False)
    if destination:
        Path(destination).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    sys.exit(main())
