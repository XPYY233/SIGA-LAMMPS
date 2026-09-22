"""R — the LAMMPS retrieval index.

Backend: **sparse BM25** over three collections (examples, docs, syntax) built
from a LAMMPS source checkout.

The paper used ChromaDB with dense embeddings. That is not what runs here, and
the reason is worth stating plainly rather than buried, because it bounds what R
can claim in the ablation.

ChromaDB's default embedder downloads an 83 MB ONNX archive on first use. In this
environment that transfer sustains roughly 20 KB/s and stalls partway; worse,
ChromaDB re-downloads whenever the archive fails its SHA256 check, so *every*
embedding call paid the failed download again. Indexing never completed a single
batch, and the symptom looks like slow embedding rather than an absent model.

A local hashing vectorizer was tried as the fallback and rejected on measurement:
~431,000 distinct features across the corpus, hashed into 1024 dimensions, is
~421 features per bucket. Rare discriminative terms drown in collisions — the
literal phrase "mean square displacement" ranked behind a timing utility. See
`bm25.py` for the full account.

So R runs on BM25: exact term statistics, no hashing, no collisions, no network,
deterministic, and auditable end to end.

**What this costs.** BM25 matches lexically. It answers command-vocabulary
queries well, which is most of what an agent asks. It does **not** answer
paraphrase — "stretch a box along one axis" shares no vocabulary with
``fix deform``. The paper's framing of R is precisely about the case where the
agent does not know the right term, so this is a genuine shortfall against the
paper, not a neutral implementation choice. `build()` records the backend in the
index so a run can always be attributed to what actually served it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapter.retrieval.bm25 import BM25Index
from adapter.retrieval.corpora import COLLECTIONS, iter_documents

__all__ = ["BuildStats", "IndexNotBuiltError", "LammpsIndex", "SearchHit"]

#: Filename of the built index inside the persist directory.
INDEX_FILENAME = "bm25.json"


class IndexNotBuiltError(RuntimeError):
    """The index is absent, so retrieval cannot answer anything."""


@dataclass(frozen=True)
class BuildStats:
    """What a build actually did, per collection."""

    per_collection: dict[str, int]
    backend: str = "bm25"

    @property
    def total(self) -> int:
        return sum(self.per_collection.values())

    def render(self) -> str:
        lines = [
            f"  {name:<10} {count:>6} documents" for name, count in sorted(self.per_collection.items())
        ]
        return "\n".join([*lines, f"  {'TOTAL':<10} {self.total:>6} documents"])


@dataclass(frozen=True)
class SearchHit:
    """One retrieval result, shaped for both the agent and a human reader."""

    doc_id: str
    collection: str
    text: str
    metadata: dict[str, Any]
    score: float

    def to_result(self) -> dict[str, Any]:
        """The model-facing shape required by `search_lammps`."""
        return {
            "source": self.metadata.get("rel_path", self.metadata.get("path", self.doc_id)),
            "collection": self.collection,
            "snippet": self.text,
            "command": self.metadata.get("command"),
            "score": round(self.score, 4),
            "metadata": {k: v for k, v in self.metadata.items() if k != "path"},
        }


class LammpsIndex:
    """A persistent BM25 index over the three LAMMPS collections."""

    def __init__(
        self,
        persist_dir: Path | str,
        *,
        model_cache_dir: Path | str | None = None,
        embedding_function: Any = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        # Retained for interface compatibility with the dense design; unused by
        # BM25, which has no model and therefore no cache.
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir is not None else None
        self._embedding_function = embedding_function
        self._index: BM25Index | None = None

    # -- locations ---------------------------------------------------------- #

    @property
    def index_path(self) -> Path:
        return self.persist_dir / INDEX_FILENAME

    @property
    def backend(self) -> str:
        """Which retrieval backend serves queries.

        Recorded per run. A retrieval result is not comparable across backends,
        and the ablation depends on knowing which one produced a hit.
        """
        return "bm25"

    # -- build -------------------------------------------------------------- #

    def build(
        self,
        root: Path | str,
        collections: Sequence[str] = COLLECTIONS,
        *,
        reset: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> BuildStats:
        """Build the index from a LAMMPS checkout.

        Args:
            root: a LAMMPS source checkout.
            collections: which collections to include.
            reset: replace the existing index. ``False`` merges with what is
                already stored, which is useful when adding one collection.
            progress: optional line-oriented progress callback.

        Returns:
            Per-collection document counts.

        Raises:
            ValueError: an unknown collection was requested.
        """
        root = Path(root)
        unknown = [name for name in collections if name not in COLLECTIONS]
        if unknown:
            raise ValueError(f"unknown collections {unknown}; expected {COLLECTIONS}")

        documents: list[dict[str, Any]] = []
        if not reset and self.index_path.is_file():
            documents = list(BM25Index.load(self.index_path).documents)
            documents = [d for d in documents if d["collection"] not in set(collections)]

        counts: dict[str, int] = {}
        for name in collections:
            found = list(iter_documents(root, [name]))
            counts[name] = len(found)
            documents.extend(
                {
                    "doc_id": d.doc_id,
                    "collection": d.collection,
                    "text": d.text,
                    "metadata": _jsonable(d.metadata),
                }
                for d in found
            )
            if progress:
                progress(f"  {name}: {len(found)} documents")

        index = BM25Index.build(documents)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        index.save(self.index_path)
        self._index = index
        return BuildStats(per_collection=counts)

    # -- load --------------------------------------------------------------- #

    def _require_index(self) -> BM25Index:
        if self._index is None:
            if not self.index_path.is_file():
                raise IndexNotBuiltError(
                    f"the retrieval index at {self.index_path} is missing or empty. Build it with:\n"
                    "  python -m adapter.cli build-index"
                )
            self._index = BM25Index.load(self.index_path)
        if len(self._index) == 0:
            raise IndexNotBuiltError(
                f"the retrieval index at {self.index_path} contains no documents. Rebuild it with:\n"
                "  python -m adapter.cli build-index"
            )
        return self._index

    def count(self) -> dict[str, int]:
        """Documents per collection. An absent index reports zero everywhere."""
        counts = {name: 0 for name in COLLECTIONS}
        if not self.index_path.is_file():
            return counts
        try:
            index = self._require_index()
        except IndexNotBuiltError:
            return counts
        for document in index.documents:
            name = document["collection"]
            counts[name] = counts.get(name, 0) + 1
        return counts

    def is_built(self) -> bool:
        return sum(self.count().values()) > 0

    def require_built(self) -> None:
        self._require_index()

    # -- query -------------------------------------------------------------- #

    def search(
        self,
        query: str,
        k: int = 5,
        collections: Iterable[str] | None = None,
    ) -> list[SearchHit]:
        """Return the *k* best passages across the requested collections.

        Results are merged and re-ranked across collections, because a caller
        asking about ``fix nvt`` wants the best answer wherever it lives — not
        five results from whichever collection happened to be queried first.

        Raises:
            ValueError: the query is empty, *k* is not positive, or an unknown
                collection was named.
            IndexNotBuiltError: nothing has been indexed.
        """
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if k <= 0:
            raise ValueError("k must be positive")

        index = self._require_index()
        wanted = list(collections) if collections is not None else list(COLLECTIONS)
        unknown = [c for c in wanted if c not in COLLECTIONS]
        if unknown:
            raise ValueError(f"unknown collections {unknown}; expected {COLLECTIONS}")
        allowed = set(wanted)

        # Over-fetch, then filter by collection. Filtering after ranking keeps a
        # single global ordering, so restricting to one collection cannot change
        # the relative order of the hits that remain.
        ranked = index.search(query, k=max(k * 4, k + 20))
        hits: list[SearchHit] = []
        for position, score in ranked:
            document = index.documents[position]
            if document["collection"] not in allowed:
                continue
            hits.append(
                SearchHit(
                    doc_id=document["doc_id"],
                    collection=document["collection"],
                    text=document["text"],
                    metadata=dict(document["metadata"]),
                    score=score,
                )
            )
            if len(hits) >= k:
                break
        return hits


def _jsonable(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only JSON-serialisable scalars, so the index round-trips."""
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif value is None:
            continue
        else:
            out[key] = str(value)
    return out
