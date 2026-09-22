"""Tests for R — corpus discovery, chunking, and RST extraction.

Unit tests run against a small synthetic corpus so they are fast, deterministic,
and independent of whether a LAMMPS checkout happens to exist on the machine.
Tests that need the real corpus are marked and skip cleanly when it is absent.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from adapter.retrieval.corpora import (
    COLLECTIONS,
    chunk_text,
    iter_documents,
    iter_docs,
    iter_examples,
    iter_syntax,
    parse_rst_page,
    strip_rst,
)
from config.loader import load_settings

# --------------------------------------------------------------------------- #
# a synthetic corpus with the same shape as a LAMMPS checkout
# --------------------------------------------------------------------------- #

_SCRIPT = """\
# a minimal script
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 4 0 4 0 4
create_box      1 box
create_atoms    1 box
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
fix             1 all nve
run             100
"""

# A command page: title, index directive, syntax block, example block, prose.
#
# Sections are underlined with `-` rather than `"` on purpose: a literal `"""`
# inside a triple-quoted Python string closes it, which is a trap worth avoiding
# in a fixture. The parser accepts any of the RST underline characters.
_COMMAND_PAGE = """\
.. index:: pair_style lj/cut

pair_style lj/cut command
=========================

Syntax
------

.. code-block:: LAMMPS

   pair_style lj/cut cutoff
   pair_style lj/cut cutoff shift

Examples
--------

.. code-block:: LAMMPS

   pair_style lj/cut 2.5
   pair_coeff 1 1 1.0 1.0

Description
-----------

The ``lj/cut`` style computes the standard 12/6 Lennard-Jones potential.

.. note::

   Cutting the potential introduces a discontinuity.
"""

# A prose page with no code blocks at all.
_PROSE_PAGE = """\
Concepts
========

About units
-----------

LAMMPS has several unit styles. Each style defines the meaning of every
physical quantity in an input script, so mixing numbers between styles is a
silent error rather than a syntax error.

The choice of style also fixes the natural timestep magnitude.
"""


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """A miniature LAMMPS checkout: examples/ plus doc/src/."""
    examples = tmp_path / "examples" / "melt"
    examples.mkdir(parents=True)
    (examples / "in.melt").write_text(_SCRIPT, encoding="utf-8")

    # A second example directory, to prove the `topic` metadata varies.
    (tmp_path / "examples" / "indent").mkdir(parents=True)
    (tmp_path / "examples" / "indent" / "in.indent").write_text(
        "units metal\nrun 10\n", encoding="utf-8"
    )

    doc_src = tmp_path / "doc" / "src"
    doc_src.mkdir(parents=True)
    (doc_src / "pair_lj_cut.rst").write_text(_COMMAND_PAGE, encoding="utf-8")
    (doc_src / "units.rst").write_text(_PROSE_PAGE, encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #


def test_short_text_is_one_chunk() -> None:
    assert chunk_text("a short rule", chunk_chars=100, overlap=10) == ["a short rule"]


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("   \n\n  ") == []


def test_long_text_splits_and_preserves_all_content() -> None:
    paragraph = "x" * 500
    text = "\n\n".join([paragraph] * 10)
    chunks = chunk_text(text, chunk_chars=1200, overlap=100)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 1200, "a chunk exceeded the target size"


def test_chunks_overlap_so_a_straddling_rule_stays_retrievable() -> None:
    paragraphs = [f"paragraph number {i} " + "y" * 300 for i in range(6)]
    chunks = chunk_text("\n\n".join(paragraphs), chunk_chars=800, overlap=150)
    assert len(chunks) >= 2
    # Consecutive chunks must share material, or a rule on the boundary is lost.
    shared = set(chunks[0][-150:].split()) & set(chunks[1][:400].split())
    assert shared, "consecutive chunks share no text; overlap is not working"


def test_oversized_single_paragraph_is_hard_split() -> None:
    chunks = chunk_text("z" * 5000, chunk_chars=1000, overlap=100)
    assert len(chunks) >= 5
    assert all(len(c) <= 1000 for c in chunks)


@pytest.mark.parametrize(("chunk_chars", "overlap"), [(0, 0), (100, 100), (100, 200), (100, -1)])
def test_invalid_chunking_parameters_are_rejected(chunk_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_chars=chunk_chars, overlap=overlap)


# --------------------------------------------------------------------------- #
# RST parsing
# --------------------------------------------------------------------------- #


def test_parse_rst_page_extracts_title_command_and_blocks() -> None:
    page = parse_rst_page(_COMMAND_PAGE)
    assert page.title == "pair_style lj/cut command"
    assert page.command == "pair_style lj/cut"
    assert len(page.code_blocks) == 2
    # First block is the syntax; it must carry the argument form, not an example.
    assert "cutoff" in page.code_blocks[0][1]
    assert page.code_blocks[0][1].startswith("pair_style lj/cut cutoff")
    # Second block is an example, with real numbers.
    assert "2.5" in page.code_blocks[1][1]


def test_parse_rst_page_handles_a_page_with_no_code_blocks() -> None:
    page = parse_rst_page(_PROSE_PAGE)
    assert page.title == "Concepts"
    assert page.code_blocks == ()


def test_code_block_bodies_lose_their_indentation() -> None:
    page = parse_rst_page(_COMMAND_PAGE)
    syntax = page.code_blocks[0][1]
    assert not syntax.startswith(" "), "indentation was not stripped from the code block"


def test_strip_rst_unwraps_inline_markup() -> None:
    text = "The ``lj/cut`` style uses **strong** emphasis and :doc:`units`."
    out = strip_rst(text)
    assert "``" not in out
    assert "**" not in out
    assert "lj/cut" in out
    assert "strong" in out
    assert "units" in out


def test_strip_rst_keeps_directive_arguments_and_drops_markers() -> None:
    out = strip_rst(_COMMAND_PAGE)
    assert "code-block" not in out, "a directive marker leaked into the prose"
    assert "index::" not in out


def test_strip_rst_removes_heading_underline_runs() -> None:
    out = strip_rst(_PROSE_PAGE)
    assert "=========" not in out
    assert "Concepts" in out


# --------------------------------------------------------------------------- #
# collection builders
# --------------------------------------------------------------------------- #


def test_iter_examples_finds_scripts_with_topic_metadata(corpus: Path) -> None:
    documents = list(iter_examples(corpus))
    assert len(documents) == 2
    topics = {d.metadata["topic"] for d in documents}
    assert topics == {"melt", "indent"}
    for document in documents:
        assert document.collection == "examples"
        assert document.metadata["kind"] == "script"
        assert document.metadata["rel_path"].startswith("examples/")


def test_example_ids_are_deterministic(corpus: Path) -> None:
    """A rebuild must be idempotent, so ids may not depend on iteration order."""
    first = {d.doc_id for d in iter_examples(corpus)}
    second = {d.doc_id for d in iter_examples(corpus)}
    assert first == second
    assert len(first) == 2


def test_iter_syntax_extracts_command_signatures(corpus: Path) -> None:
    documents = list(iter_syntax(corpus))
    assert len(documents) == 1, "only the command page carries a syntax block"
    document = documents[0]
    assert document.collection == "syntax"
    assert document.metadata["command"] == "pair_style lj/cut"
    assert "pair_style lj/cut cutoff" in document.text
    # The example is folded in, because syntax alone does not show usage.
    assert "# example" in document.text


def test_iter_docs_chunks_prose_and_skips_bare_command_stubs(corpus: Path) -> None:
    documents = list(iter_docs(corpus))
    assert documents, "the prose page must produce documents"
    for document in documents:
        assert document.collection == "docs"
        assert document.metadata["kind"] == "doc"
    sources = {d.metadata["rel_path"] for d in documents}
    # The prose page is indexed...
    assert "doc/src/units.rst" in sources
    # ...and the short command stub is not, since `syntax` already covers it.
    assert "doc/src/pair_lj_cut.rst" not in sources


def test_iter_documents_rejects_an_unknown_collection(corpus: Path) -> None:
    with pytest.raises(ValueError, match="unknown collection"):
        list(iter_documents(corpus, ["nope"]))


def test_iter_documents_covers_every_collection(corpus: Path) -> None:
    seen = {d.collection for d in iter_documents(corpus, COLLECTIONS)}
    assert seen == set(COLLECTIONS)


def test_missing_subdirectories_yield_nothing_instead_of_raising(tmp_path: Path) -> None:
    """A partial checkout must degrade, not explode."""
    assert list(iter_examples(tmp_path)) == []
    assert list(iter_docs(tmp_path)) == []
    assert list(iter_syntax(tmp_path)) == []


# --------------------------------------------------------------------------- #
# the real corpus — skipped unless a LAMMPS checkout is configured
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_corpus() -> Path:
    from config.loader import load_settings

    settings = load_settings()
    root = settings.retrieval.corpus_dir
    if not (root / "examples").is_dir() or not (root / "doc" / "src").is_dir():
        pytest.skip(f"no LAMMPS checkout at {root}")
    return root


@pytest.mark.real_corpus
def test_real_corpus_yields_a_substantial_index(real_corpus: Path) -> None:
    counts = {name: sum(1 for _ in iter_documents(real_corpus, [name])) for name in COLLECTIONS}
    assert counts["examples"] > 100, f"too few example scripts: {counts}"
    assert counts["docs"] > 100, f"too few doc chunks: {counts}"
    assert counts["syntax"] > 100, f"too few syntax pages: {counts}"


@pytest.mark.real_corpus
def test_real_corpus_contains_the_commands_our_tasks_need(real_corpus: Path) -> None:
    """R must be able to answer for every command the five tasks depend on.

    If these are missing, X would flag a script that R cannot help repair — the
    worst combination, since the agent would be told it is wrong with no way to
    find out what right looks like.
    """
    commands = {d.metadata.get("command", "") for d in iter_syntax(real_corpus)}
    joined = " | ".join(sorted(commands))
    for required in ("compute msd", "fix deform", "fix indent", "pair_style lj/cut", "fix nvt/sllod"):
        assert required in joined, f"the syntax collection has no entry for {required!r}"


@pytest.mark.real_corpus
def test_real_syntax_entries_carry_usable_signatures(real_corpus: Path) -> None:
    documents = list(iter_syntax(real_corpus))
    with_command = [d for d in documents if d.metadata.get("command")]
    assert len(with_command) == len(documents), "every syntax entry must name a command"
    # A signature is only useful if it shows arguments, not just a bare keyword.
    multi_word = sum(1 for d in documents if " " in d.text.split("\n")[2])
    assert multi_word > len(documents) * 0.5


# --------------------------------------------------------------------------- #
# search quality — needs a built index; skips when absent
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def built_index() -> Any:
    from adapter.retrieval.index import LammpsIndex

    settings = load_settings()
    index = LammpsIndex(
        settings.retrieval.persist_dir,
        model_cache_dir=settings.retrieval.model_cache_dir,
    )
    if sum(index.count().values()) == 0:
        pytest.skip("retrieval index not built; run `python -m adapter.cli build-index`")
    return index


@pytest.mark.slow
@pytest.mark.parametrize(
    ("query", "expected_source_fragment"),
    [
        ("compute msd", "compute_msd"),
        ("fix deform", "fix_deform"),
        ("fix indent", "fix_indent"),
        ("fix nvt/sllod", "fix_nvt_sllod"),
        ("pair_style lj/cut", "pair_lj_cut"),
        ("run", "run"),
    ],
)
def test_search_answers_command_vocabulary(
    built_index: Any, query: str, expected_source_fragment: str
) -> None:
    """R must answer queries phrased in LAMMPS vocabulary.

    This is the common case and BM25's strength: an agent that already knows it
    needs ``fix deform`` is asking for the syntax, not for the concept.
    """
    hits = built_index.search(query, k=5)
    assert hits, f"no results for {query!r}"
    sources = " ".join(h.metadata.get("rel_path", "") for h in hits)
    assert expected_source_fragment in sources, (
        f"query {query!r} did not surface {expected_source_fragment!r} in its top "
        f"{len(hits)}; got {sources}"
    )


#: Paraphrases BM25 cannot reach, recorded rather than deleted.
#:
#: `strict=True` is the point: if one of these starts passing, the suite fails and
#: says so, so the limitation is revisited instead of quietly forgotten. If one
#: regresses further that also shows up, because the body still asserts.
#:
#: These are the queries the paper says R exists for — "when the agent does not
#: know the right simulator terms to search for". Closing them needs dense
#: embeddings; see `adapter/retrieval/index.py` for why that is not the backend
#: here.
KNOWN_PARAPHRASE_LIMITS = [
    pytest.param(
        "stretch a box along one axis",
        "fix_deform",
        marks=pytest.mark.xfail(strict=True, reason="no lexical overlap with 'fix deform'"),
    ),
    pytest.param(
        "thermostat to hold a constant temperature",
        "nvt",
        marks=pytest.mark.xfail(strict=True, reason="'thermostat' does not appear on the nvt page verbatim"),
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(("query", "expected_source_fragment"), KNOWN_PARAPHRASE_LIMITS)
def test_search_paraphrase_is_a_known_gap(
    built_index: Any, query: str, expected_source_fragment: str
) -> None:
    """A known, measured shortfall against the paper's intent for R."""
    hits = built_index.search(query, k=5)
    sources = " ".join(h.metadata.get("rel_path", "") for h in hits)
    assert expected_source_fragment in sources


@pytest.mark.slow
def test_search_does_answer_some_paraphrase(built_index: Any) -> None:
    """Not every paraphrase fails, so the gap is bounded rather than total.

    An indentation query reaches `fix_indent` because the shared vocabulary
    ("indenter", "sphere") happens to be present. This keeps the xfail list
    above honest: it is a list of specific misses, not a blanket excuse.
    """
    hits = built_index.search("spherical indenter pressing into a surface", k=5)
    sources = " ".join(h.metadata.get("rel_path", "") for h in hits)
    assert "fix_indent" in sources


@pytest.mark.slow
def test_search_respects_k(built_index: Any) -> None:
    assert len(built_index.search("fix nvt", k=3)) <= 3
    assert len(built_index.search("fix nvt", k=1)) == 1


@pytest.mark.slow
def test_search_can_be_restricted_to_one_collection(built_index: Any) -> None:
    hits = built_index.search("fix deform", k=5, collections=["syntax"])
    assert hits
    assert {h.collection for h in hits} == {"syntax"}


@pytest.mark.slow
def test_search_scores_are_ordered_and_normalised(built_index: Any) -> None:
    hits = built_index.search("lennard-jones cutoff", k=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), "results must be ranked by score"
    assert all(-1.0 <= s <= 1.0 for s in scores), f"score outside cosine range: {scores}"


@pytest.mark.slow
def test_search_rejects_empty_query(built_index: Any) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        built_index.search("   ")


@pytest.mark.slow
def test_search_rejects_unknown_collection(built_index: Any) -> None:
    with pytest.raises(ValueError, match="unknown collections"):
        built_index.search("fix nvt", collections=["bogus"])


@pytest.mark.slow
def test_hits_carry_the_fields_the_tool_contract_promises(built_index: Any) -> None:
    """`search_lammps` must return source, snippet, command and metadata."""
    hit = built_index.search("pair_style lj/cut", k=1)[0].to_result()
    assert set(hit) >= {"source", "collection", "snippet", "command", "score", "metadata"}
    assert hit["source"], "a hit without a citable source is not checkable"
    assert hit["snippet"].strip(), "a hit without a snippet is useless"


@pytest.mark.slow
def test_examples_collection_returns_runnable_scripts(built_index: Any) -> None:
    """A hit from `examples` must look like a script, not prose about one."""
    hits = built_index.search("lennard-jones melt fcc lattice", k=5, collections=["examples"])
    assert hits, "the examples collection returned nothing for a core MD query"
    best = hits[0].text
    # Assert it is an input script, not that it is the *ideal* one. The corpus
    # holds many variants of each example (MDI, GPU, accelerated), and several
    # legitimately omit a plain `run` in favour of a driver interface.
    assert "units" in best and "atom_style" in best, (
        "the top example hit does not look like an input script; the examples "
        "collection may be indexing README files instead of in.* scripts"
    )
