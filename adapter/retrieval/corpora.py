"""Corpus discovery for R — the LAMMPS knowledge base.

Three collections, mirroring the paper's LAMMPS port:

``examples``
    Real, runnable LAMMPS input scripts. The most valuable collection: a
    retrieval hit here is a working script, not a description of one.
``docs``
    The reStructuredText manual, chunked. Prose explanation — why a command
    behaves as it does.
``syntax``
    Extracted command signatures, one per manual page. Precise argument forms,
    which is what stops an agent inventing a keyword.

Everything is derived from a LAMMPS source checkout. Nothing is downloaded and
nothing is committed: ``data/raw`` points at the checkout and the index is
rebuilt from it.

Document ids are deterministic hashes of (collection, path, chunk), so rebuilding
an unchanged corpus is idempotent and a hit can always be traced back to a file.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "COLLECTIONS",
    "Document",
    "chunk_text",
    "iter_documents",
    "iter_docs",
    "iter_examples",
    "iter_syntax",
    "lammps_root",
]

COLLECTIONS = ("examples", "docs", "syntax")

#: Target chunk size in characters. Chosen to sit inside the embedding model's
#: window while staying long enough to carry a complete rule.
CHUNK_CHARS = 1800
CHUNK_OVERLAP = 200

#: LAMMPS input scripts are usually tiny; a script longer than this is split.
SCRIPT_CHUNK_CHARS = 2400

_RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")
_CODE_BLOCK = re.compile(r"^(\s*)\.\.\s+code-block::\s*(\S+)", re.IGNORECASE)
_INDEX_DIRECTIVE = re.compile(r"^\.\.\s+index::\s*(.+)$", re.IGNORECASE)
_DIRECTIVE = re.compile(r"^\.\.\s+[a-zA-Z-]+::")


@dataclass(frozen=True)
class Document:
    """One indexable unit, with everything a search hit needs to be judged."""

    doc_id: str
    collection: str
    text: str
    #: `source` is what the caller cites; `path` is absolute for local tooling.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_result(self) -> dict[str, Any]:
        """The model-facing shape required by `search_lammps`."""
        return {
            "source": self.metadata.get("rel_path", self.metadata.get("path", self.doc_id)),
            "collection": self.collection,
            "snippet": self.text,
            "command": self.metadata.get("command"),
            "metadata": dict(self.metadata),
        }


def _doc_id(collection: str, rel_path: str, index: int) -> str:
    digest = hashlib.sha256(f"{collection}\0{rel_path}\0{index}".encode()).hexdigest()
    return f"{collection}:{digest[:16]}"


def lammps_root(configured: Path | str | None = None) -> Path:
    """Resolve the LAMMPS source checkout to read the corpus from.

    Raises:
        FileNotFoundError: the root or its expected subdirectories are missing.
    """
    if configured is None:
        raise FileNotFoundError("no LAMMPS corpus root configured (set retrieval.corpus_dir)")
    root = Path(configured)
    if not root.is_dir():
        raise FileNotFoundError(
            f"LAMMPS corpus root not found: {root}\n"
            "Point retrieval.corpus_dir in config/config.yaml at a LAMMPS source "
            "checkout (it must contain examples/ and doc/src/)."
        )
    return root


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #


def chunk_text(
    text: str,
    *,
    chunk_chars: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping chunks, preferring paragraph boundaries.

    Splitting mid-sentence loses the subject; splitting on a blank line keeps a
    rule intact. Overlap exists so a rule straddling a boundary is still
    retrievable from one side.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap < 0 or overlap >= chunk_chars:
        raise ValueError("overlap must be >= 0 and < chunk_chars")

    cleaned = _normalise(text)
    if not cleaned:
        return []
    if len(cleaned) <= chunk_chars:
        return [cleaned]

    paragraphs = [p for p in cleaned.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        # A single oversized paragraph is hard-split; there is no better boundary.
        if len(paragraph) > chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(paragraph), chunk_chars - overlap):
                chunks.append(paragraph[start : start + chunk_chars])
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > chunk_chars:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{paragraph}" if tail else paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


def _normalise(text: str) -> str:
    """Light cleaning: collapse blank runs, strip trailing space, keep structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


# --------------------------------------------------------------------------- #
# examples
# --------------------------------------------------------------------------- #


def iter_examples(root: Path) -> Iterator[Document]:
    """Yield one document per LAMMPS input script found under ``examples/``."""
    base = root / "examples" if (root / "examples").is_dir() else root
    for path in sorted(base.rglob("in.*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.strip():
            continue
        rel = path.relative_to(root).as_posix()
        # The example directory is a useful topical signal: `examples/indent`
        # tells the retriever more than the filename does.
        topic = path.parent.name
        chunks = chunk_text(text, chunk_chars=SCRIPT_CHUNK_CHARS, overlap=0)
        for index, chunk in enumerate(chunks):
            yield Document(
                doc_id=_doc_id("examples", rel, index),
                collection="examples",
                text=chunk,
                metadata={
                    "path": str(path),
                    "rel_path": rel,
                    "topic": topic,
                    "kind": "script",
                    "chunk": index,
                    "chunks": len(chunks),
                },
            )


# --------------------------------------------------------------------------- #
# syntax
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _RstPage:
    title: str
    command: str | None
    code_blocks: tuple[tuple[str, str], ...]  # (language, body)


def parse_rst_page(text: str) -> _RstPage:
    """Extract a manual page's title, indexed command, and code blocks.

    LAMMPS manual command pages share a rigid shape: a title underlined with
    ``=``, an ``.. index::`` naming the command, then `code-block` directives
    whose first occurrence is the syntax and whose later ones are examples.
    """
    lines = text.split("\n")

    title = ""
    for i in range(len(lines) - 1):
        candidate = lines[i].strip()
        if candidate and _RST_UNDERLINE.match(lines[i + 1]) and lines[i + 1].startswith("="):
            title = candidate
            break

    command: str | None = None
    match = _INDEX_DIRECTIVE.match(text.lstrip().split("\n", 1)[0]) if text.strip() else None
    for line in lines:
        found = _INDEX_DIRECTIVE.match(line.strip())
        if found:
            command = found.group(1).strip()
            break

    code_blocks: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        block = _CODE_BLOCK.match(lines[i])
        if block is None:
            i += 1
            continue
        indent = len(block.group(1)) + 3
        body: list[str] = []
        j = i + 1
        # Skip directive options such as `:linenos:`.
        while j < len(lines) and lines[j].strip().startswith(":"):
            j += 1
        while j < len(lines):
            line = lines[j]
            if line.strip() == "":
                body.append("")
                j += 1
                continue
            if len(line) - len(line.lstrip()) < indent:
                break
            body.append(line[indent:] if len(line) >= indent else line.strip())
            j += 1
        code_blocks.append((block.group(2), "\n".join(body).strip()))
        i = j

    return _RstPage(title=title, command=command, code_blocks=tuple(code_blocks))


def iter_syntax(root: Path) -> Iterator[Document]:
    """Yield one document per manual page carrying extracted command syntax."""
    doc_src = root / "doc" / "src"
    if not doc_src.is_dir():
        return
    for path in sorted(doc_src.glob("*.rst")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        page = parse_rst_page(text)
        if not page.code_blocks:
            continue
        # First block is the syntax; later ones are examples. Keep both: the
        # syntax is what stops a wrong keyword, the example shows it in use.
        syntax = page.code_blocks[0][1]
        if not syntax.strip():
            continue
        examples = [body for _, body in page.code_blocks[1:3] if body.strip()]
        parts = [f"# {page.title or path.stem}", "", syntax]
        for example in examples:
            parts += ["", "# example", example]
        rel = path.relative_to(root).as_posix()
        yield Document(
            doc_id=_doc_id("syntax", rel, 0),
            collection="syntax",
            text="\n".join(parts),
            metadata={
                "path": str(path),
                "rel_path": rel,
                "command": page.command or page.title or path.stem,
                "title": page.title,
                "kind": "syntax",
                "chunk": 0,
                "chunks": 1,
            },
        )


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #


def iter_docs(root: Path) -> Iterator[Document]:
    """Yield chunked prose from the manual, excluding pure command-syntax pages.

    Pages already covered by the `syntax` collection are skipped, so a query
    like "fix nvt" does not return the same content twice under two names.
    """
    doc_src = root / "doc" / "src"
    if not doc_src.is_dir():
        return
    for path in sorted(doc_src.glob("*.rst")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        page = parse_rst_page(text)
        # A syntax-only command stub adds nothing beyond the `syntax` entry.
        if page.code_blocks and len(_normalise(text)) < CHUNK_CHARS * 2:
            continue
        body = strip_rst(text)
        rel = path.relative_to(root).as_posix()
        chunks = chunk_text(body)
        for index, chunk in enumerate(chunks):
            yield Document(
                doc_id=_doc_id("docs", rel, index),
                collection="docs",
                text=chunk,
                metadata={
                    "path": str(path),
                    "rel_path": rel,
                    "title": page.title,
                    "section": _first_section(text),
                    "kind": "doc",
                    "chunk": index,
                    "chunks": len(chunks),
                },
            )


def _first_section(text: str) -> str:
    """The first sub-heading, which is usually the most specific label."""
    lines = text.split("\n")
    for i in range(len(lines) - 1):
        if lines[i].strip() and _RST_UNDERLINE.match(lines[i + 1]) and lines[i + 1].startswith("-"):
            return lines[i].strip()
    return ""


def strip_rst(text: str) -> str:
    """Reduce RST to readable prose, preserving headings as context.

    Directives become their argument (so `.. code-block:: LAMMPS` leaves the
    code behind rather than a dangling marker), and inline markup is unwrapped.
    Headings are kept because a chunk without its heading loses its subject.
    """
    out: list[str] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if _CODE_BLOCK.match(line):
            # Drop the marker but KEEP the body. Leaving the marker in puts raw
            # `.. code-block:: LAMMPS` text into the embedded document, which is
            # pure noise for retrieval: it matches no user query and dilutes the
            # passage it precedes.
            i += 1
            while i < len(lines) and lines[i].strip().startswith(":"):
                i += 1  # directive options such as `:linenos:`
            continue
        if _DIRECTIVE.match(stripped):
            # Keep the argument of directives that name something useful.
            argument = stripped.split("::", 1)[1].strip() if "::" in stripped else ""
            if argument:
                out.append(argument)
            i += 1
            continue
        if _RST_UNDERLINE.match(line) and out and out[-1].strip():
            # Underline of a heading already emitted: drop the punctuation run.
            i += 1
            continue
        out.append(line)
        i += 1
    text = "\n".join(out)
    # Unwrap inline roles and emphasis markers.
    text = re.sub(r":[a-zA-Z:]+:`([^`<]*?)\s*<[^`>]*>`", r"\1", text)
    text = re.sub(r":[a-zA-Z:]+:`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"``([^`]+)``", r"\1", text)
    return _normalise(text)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def iter_documents(
    root: Path,
    collections: Iterable[str] = COLLECTIONS,
) -> Iterator[Document]:
    """Yield every document for the requested collections.

    Raises:
        ValueError: an unknown collection name was requested.
    """
    requested = list(collections)
    for name in requested:
        if name not in COLLECTIONS:
            raise ValueError(f"unknown collection {name!r}; expected one of {COLLECTIONS}")

    builders = {"examples": iter_examples, "docs": iter_docs, "syntax": iter_syntax}
    for name in requested:
        yield from builders[name](root)
