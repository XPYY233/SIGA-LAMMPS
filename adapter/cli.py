"""The single adapter entry point shared by every consumer.

The MCP server's tools, the harness stop gate (S), and the benchmark evaluator
all reach the adapter through this module. That is deliberate: the paper's LAMMPS
port describes its validator as *"applying the same checks as the hook"*, and the
only way to make that identity structural rather than a promise is for both to
call one implementation.

**Exit-code contract.** A consumer must be able to tell "the input is invalid"
from "the validator is broken", because the correct response differs: the first
means block the agent, the second means alert a human. Collapsing them is how a
broken validator silently disables the stop gate and quietly turns an M+R+X+S run
into a vanilla one.

===== ==================================================================
   0  ran successfully, input is VALID
   1  ran successfully, input is INVALID
   2  could not run — usage error, missing config, missing index, crash
===== ==================================================================

Subcommands::

    search        R — query the LAMMPS knowledge base
    build-index   R — build the index from a LAMMPS checkout
    index-status  R — report index size and location
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from config.loader import DEFAULT_CONFIG_FILE, DEFAULT_ENV_FILE, ConfigError, Settings, load_settings

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_ERROR = 2


class CliError(RuntimeError):
    """A failure to run, as opposed to a finding about the input."""


# --------------------------------------------------------------------------- #
# output helpers
# --------------------------------------------------------------------------- #


def _emit(payload: Any, *, as_json: bool, text: str = "") -> None:
    if as_json:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False, sort_keys=True)
        sys.stdout.write("\n")
    elif text:
        print(text)


def _make_index(args: argparse.Namespace, settings: Settings) -> Any:
    """Construct the index exactly one way, so every command shares cache policy."""
    from adapter.retrieval.index import LammpsIndex

    return LammpsIndex(
        args.persist_dir or settings.retrieval.persist_dir,
        model_cache_dir=settings.retrieval.model_cache_dir,
    )


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #


def cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    index = _make_index(args, settings)
    collections = args.collections.split(",") if args.collections else None

    try:
        hits = index.search(args.query, k=args.k, collections=collections)
    except Exception as exc:  # noqa: BLE001 - reported as a run failure, not a finding
        raise CliError(str(exc)) from exc

    results = [hit.to_result() for hit in hits]
    if args.json:
        _emit({"query": args.query, "count": len(results), "results": results}, as_json=True)
    else:
        if not results:
            print("no results")
        for hit in results:
            header = f"[{hit.collection}] {hit.metadata.get('rel_path', hit.doc_id)}  score={hit.score:.3f}"
            print(header)
            body = hit.text if len(hit.text) <= 600 else hit.text[:600] + " …"
            print("\n".join(f"    {line}" for line in body.splitlines()))
            print()
    return EXIT_OK


def cmd_build_index(args: argparse.Namespace, settings: Settings) -> int:
    from adapter.retrieval.corpora import COLLECTIONS

    root = Path(args.root) if args.root else settings.retrieval.corpus_dir
    if not (root / "examples").is_dir() and not (root / "doc").is_dir():
        raise CliError(
            f"{root} does not look like a LAMMPS source checkout "
            "(expected examples/ and doc/src/). Pass --root, or set "
            "SIGA_LAMMPS_CORPUS in .env."
        )

    collections = tuple(args.collections.split(",")) if args.collections else COLLECTIONS
    index = _make_index(args, settings)
    stats = index.build(
        root,
        collections,
        reset=not args.keep,
        progress=None if args.json else (lambda line: print(line, flush=True)),
    )

    if args.json:
        _emit({"root": str(root), "counts": stats.per_collection, "total": stats.total}, as_json=True)
    else:
        print(f"\nindexed {root}\n{stats.render()}\n  at {index.persist_dir}")
    return EXIT_OK


def cmd_index_status(args: argparse.Namespace, settings: Settings) -> int:
    index = _make_index(args, settings)
    try:
        counts = index.count()
    except Exception as exc:  # noqa: BLE001
        raise CliError(str(exc)) from exc

    payload = {"persist_dir": str(index.persist_dir), "counts": counts, "total": sum(counts.values())}
    if args.json:
        _emit(payload, as_json=True)
    else:
        print(f"index: {payload['persist_dir']}")
        for name, count in sorted(counts.items()):
            print(f"  {name:<10} {count:>6}")
        print(f"  {'TOTAL':<10} {payload['total']:>6}")
    return EXIT_OK if payload["total"] else EXIT_INVALID


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def _add_common_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Add the options accepted both before and after a subcommand.

    argparse normally lets a subparser's default clobber a value set on the
    parent, so `--json search ...` would be silently undone. Giving the
    subparser copy `default=SUPPRESS` means it only writes the attribute when the
    user actually passed the flag, which makes both positions work.
    """
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS if suppress else False,
        help="emit machine-readable JSON on stdout",
    )
    parser.add_argument(
        "--env-file",
        default=argparse.SUPPRESS if suppress else None,
        help="path to .env (default: the repository .env)",
    )
    parser.add_argument(
        "--no-env-file",
        action="store_true",
        default=argparse.SUPPRESS if suppress else False,
        help="ignore .env entirely and read only the process environment",
    )
    parser.add_argument(
        "--config",
        default=argparse.SUPPRESS if suppress else str(DEFAULT_CONFIG_FILE),
        help="path to config.yaml (default: the repository config/config.yaml)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adapter.cli",
        description="SIGA-LAMMPS adapter entry point (R, X).",
    )
    _add_common_flags(parser, suppress=False)
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="R: query the LAMMPS knowledge base")
    _add_common_flags(search, suppress=True)
    search.add_argument("query", help="natural-language or keyword query")
    search.add_argument("--k", type=int, default=None, help="number of results (default from config)")
    search.add_argument(
        "--collections", default=None, help="comma-separated subset of examples,docs,syntax"
    )
    search.add_argument("--persist-dir", default=None, help="override the index location")
    search.set_defaults(func=cmd_search)

    build = sub.add_parser("build-index", help="R: build the index from a LAMMPS checkout")
    _add_common_flags(build, suppress=True)
    build.add_argument("--root", default=None, help="LAMMPS source checkout (default from config)")
    build.add_argument(
        "--collections", default=None, help="comma-separated subset of examples,docs,syntax"
    )
    build.add_argument("--persist-dir", default=None, help="override the index location")
    build.add_argument(
        "--keep", action="store_true", help="upsert into the existing index instead of replacing it"
    )
    build.set_defaults(func=cmd_build_index)

    status = sub.add_parser("index-status", help="R: report index size and location")
    _add_common_flags(status, suppress=True)
    status.add_argument("--persist-dir", default=None, help="override the index location")
    status.set_defaults(func=cmd_index_status)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # `env_file=None` means "skip .env entirely" to the loader, so the CLI needs
    # a distinct way to say "use the repository default". Hence --no-env-file.
    env_file: Path | str | None
    if args.no_env_file:
        env_file = None
    elif args.env_file is not None:
        env_file = args.env_file
    else:
        env_file = DEFAULT_ENV_FILE

    try:
        settings = load_settings(env_file=env_file, config_file=args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if getattr(args, "k", None) is None:
        args.k = settings.retrieval.top_k

    try:
        return int(args.func(args, settings))
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
