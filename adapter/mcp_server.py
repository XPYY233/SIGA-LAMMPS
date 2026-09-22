"""The SIGA-LAMMPS tools, exposed over MCP stdio.

This is how the agent reaches the adapter. The harness spawns this process and
registers each tool as `mcp__<server>__<tool>`, so the adapter stays a separate
Python process and the harness never learns anything about LAMMPS.

Two decisions worth stating:

* **The server loads its own configuration.** The harness does not read our
  `.env` — its own `.env` handling is credentials-only — so relying on it would
  silently produce a server with no corpus path and no index. Instead
  `config.loader` resolves everything from the repository root, which it derives
  from this file's own location and therefore works whatever the cwd is.

* **Errors are returned as structured content, not raised.** A tool that throws
  gives the model an opaque failure; a tool that returns
  ``{"error": ..., "hint": ...}`` lets it decide whether to retry, ask, or take a
  different route. The one exception is a genuinely broken invocation, which is
  a real error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from config.loader import ConfigError, Settings, load_settings

__all__ = ["build_server", "main", "SERVER_NAME"]

SERVER_NAME = "siga-lammps"


def _settings() -> Settings:
    """Load settings on each call, so a config edit does not need a restart."""
    return load_settings()


def build_server() -> MCPServer:
    """Construct the MCP server and register every tool."""
    server = MCPServer(
        name=SERVER_NAME,
        instructions=(
            "Tools for authoring LAMMPS simulations. Use search_lammps for command "
            "syntax and worked examples, and validate_lammps_input to check a "
            "workspace before considering it finished. Validation is deterministic "
            "and its findings are authoritative."
        ),
    )

    # ----------------------------------------------------------------- R --- #
    @server.tool(
        name="search_lammps",
        description=(
            "Search the LAMMPS documentation, example scripts and command reference. "
            "Returns the most relevant passages with their source paths. Call this "
            "before writing a command whose arguments you are not certain of, and "
            "again when validation reports something you do not recognise. Prefer it "
            "over guessing a keyword: LAMMPS does not always error on a wrong "
            "argument, it sometimes silently computes something else."
        ),
    )
    def search_lammps(query: str, k: int = 5, collections: str = "") -> dict[str, Any]:
        """Query the LAMMPS knowledge base.

        Args:
            query: natural language or LAMMPS vocabulary, e.g. "fix deform erate"
                or "mean square displacement".
            k: how many passages to return.
            collections: optional comma-separated subset of examples, docs, syntax.
        """
        from adapter.retrieval.index import IndexNotBuiltError, LammpsIndex

        try:
            settings = _settings()
            index = LammpsIndex(
                settings.retrieval.persist_dir,
                model_cache_dir=settings.retrieval.model_cache_dir,
            )
            wanted = [c.strip() for c in collections.split(",") if c.strip()] or None
            hits = index.search(query, k=k, collections=wanted)
        except IndexNotBuiltError as exc:
            return {
                "error": "index_not_built",
                "detail": str(exc),
                "hint": "The retrieval index has not been built. Report this; do not guess syntax.",
                "results": [],
            }
        except (ValueError, ConfigError) as exc:
            return {"error": "bad_request", "detail": str(exc), "results": []}

        return {
            "query": query,
            "backend": index.backend,
            "count": len(hits),
            "results": [hit.to_result() for hit in hits],
            "note": (
                "Retrieval matches LAMMPS vocabulary well; a paraphrase may return "
                "nothing useful. If these results do not answer the question, say so "
                "rather than inventing syntax."
            ),
        }

    # ----------------------------------------------------------------- X --- #
    @server.tool(
        name="validate_lammps_input",
        description=(
            "Deterministically validate a LAMMPS workspace: required files, command "
            "ordering, units, atom_style, structure initialisation, force field, "
            "referenced files, ensemble, timestep, run, and obvious conflicts. "
            "Call it before considering a task finished. Findings are authoritative "
            "for structure but say nothing about whether the physics is right."
        ),
    )
    def validate_lammps_input(workspace: str = ".", task: str = "") -> dict[str, Any]:
        """Validate the LAMMPS input script in a workspace.

        Args:
            workspace: directory containing the input script. Relative paths
                resolve against the current working directory.
            task: optional benchmark task id to validate against, which adds that
                task's required commands and patterns to the checks.
        """
        from adapter.tasks import TaskSpecError, load_task
        from adapter.validator import validate_workspace

        try:
            settings = _settings()
            path = Path(workspace).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()

            spec = load_task(task, settings.benchmark.tasks_dir) if task else None
            result = validate_workspace(
                path,
                task=spec,
                supported_atom_styles=settings.validator.supported_atom_styles,
                supported_unit_styles=settings.validator.supported_unit_styles,
                max_lines=settings.validator.max_lines,
            )
        except FileNotFoundError as exc:
            # Nothing to validate is not an invalid input.
            return {"error": "no_input_script", "detail": str(exc), "valid": None}
        except TaskSpecError as exc:
            return {"error": "unknown_task", "detail": str(exc), "valid": None}
        except ConfigError as exc:
            return {"error": "config", "detail": str(exc), "valid": None}

        payload = result.to_dict()
        payload["note"] = (
            "Only errors block. Warnings and suggestions are worth reading but do not "
            "make the script invalid."
        )
        return payload

    return server


def main() -> None:
    """Run the server over stdio, as the harness expects."""
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
