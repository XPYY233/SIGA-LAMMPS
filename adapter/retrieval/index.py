"""R — the ChromaDB-backed LAMMPS index.

Three collections (examples, docs, syntax), each embedded locally with
ChromaDB's default ONNX model. **No API key and no per-query network call**: the
embedding model is fetched once, then retrieval is entirely local. That matters
because R runs inside a tool the agent may call many times per task, and a
network round trip per call would dominate both latency and cost.

Builds are idempotent — document ids are deterministic hashes, and documents are
upserted — so re-running a build over an unchanged corpus is cheap and safe.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapter.retrieval.corpora import COLLECTIONS, Document, iter_documents

__all__ = ["BuildStats", "LammpsIndex", "SearchHit", "IndexNotBuiltError"]

#: ChromaDB degrades above a few thousand records per call; batch conservatively.
BATCH_SIZE = 512


class IndexNotBuiltError(RuntimeError):
    """The index is absent, so retrieval cannot answer anything."""


@dataclass(frozen=True)
class BuildStats:
    """What a build actually did, per collection."""

    per_collection: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.per_collection.values())

    def render(self) -> str:
        lines = [f"  {name:<10} {count:>6} documents" for name, count in sorted(self.per_collection.items())]
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


def _require_chromadb() -> Any:
    try:
        import chromadb  # noqa: PLC0415 - optional heavy dependency
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "chromadb is required for retrieval. Install it with:\n"
            "  .venv/bin/python -m pip install chromadb"
        ) from exc
    return chromadb


class LammpsIndex:
    """A persistent, locally-embedded index over the three LAMMPS collections."""

    def __init__(
        self,
        persist_dir: Path | str,
        *,
        model_cache_dir: Path | str | None = None,
        embedding_function: Any = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir is not None else None
        self._embedding_function = embedding_function
        self._client: Any = None

    # -- lifecycle ---------------------------------------------------------- #

    def _resolve_embedding_function(self) -> Any:
        """Return the embedding function, redirecting ChromaDB's model cache.

        ChromaDB's ONNX MiniLM writes to ``~/.cache/chroma`` by default — a plain
        class attribute with no environment override. Two reasons to move it:

        * It puts several hundred MB in the user's home for a project artifact.
        * Under a sandboxed filesystem, writing to ``$HOME`` is denied outright,
          so the default simply fails with a permissions error that looks
          nothing like its cause.

        Pinning it inside the project makes the cache reproducible and removable
        alongside everything else in ``data/``.
        """
        if self._embedding_function is not None:
            return self._embedding_function
        if self.model_cache_dir is None:
            return None

        try:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        except ImportError:  # pragma: no cover - chromadb absent
            return None

        download_path = self.model_cache_dir / "onnx_models" / ONNXMiniLM_L6_V2.MODEL_NAME
        download_path.mkdir(parents=True, exist_ok=True)
        ONNXMiniLM_L6_V2.DOWNLOAD_PATH = download_path
        return ONNXMiniLM_L6_V2()

    def _require_client(self) -> Any:
        if self._client is None:
            chromadb = _require_chromadb()
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    def _collection(self, name: str) -> Any:
        client = self._require_client()
        embedding = self._resolve_embedding_function()
        if embedding is not None:
            return client.get_or_create_collection(
                name=name,
                embedding_function=embedding,
                metadata={"hnsw:space": "cosine"},
            )
        return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    # -- build -------------------------------------------------------------- #

    def build(
        self,
        root: Path | str,
        collections: Sequence[str] = COLLECTIONS,
        *,
        reset: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> BuildStats:
        """Embed and store the corpus under *root*.

        Args:
            root: a LAMMPS source checkout.
            collections: which collections to build.
            reset: drop each collection first, so a build is a full replacement
                rather than a merge with stale documents.
            progress: optional line-oriented progress callback.

        Returns:
            Per-collection document counts.
        """
        root = Path(root)
        client = self._require_client()
        stats: dict[str, int] = {}

        for name in collections:
            if reset:
                try:
                    client.delete_collection(name)
                except Exception:  # noqa: BLE001 - absence is the common case
                    pass
            collection = self._collection(name)

            batch: list[Document] = []
            written = 0
            for document in iter_documents(root, [name]):
                batch.append(document)
                if len(batch) >= BATCH_SIZE:
                    written += self._flush(collection, batch)
                    batch = []
                    if progress:
                        progress(f"  {name}: {written} documents")
            if batch:
                written += self._flush(collection, batch)

            stats[name] = written
            if progress:
                progress(f"  {name}: {written} documents (done)")

        return BuildStats(per_collection=stats)

    @staticmethod
    def _flush(collection: Any, batch: list[Document]) -> int:
        collection.upsert(
            ids=[d.doc_id for d in batch],
            documents=[d.text for d in batch],
            metadatas=[_jsonable(d.metadata) for d in batch],
        )
        return len(batch)

    # -- query -------------------------------------------------------------- #

    def count(self) -> dict[str, int]:
        """Documents per collection. Absent collections report zero."""
        client = self._require_client()
        available = {c.name for c in client.list_collections()}
        return {
            name: (self._collection(name).count() if name in available else 0)
            for name in COLLECTIONS
        }

    def is_built(self) -> bool:
        return sum(self.count().values()) > 0

    def require_built(self) -> None:
        counts = self.count()
        if sum(counts.values()) == 0:
            raise IndexNotBuiltError(
                f"the retrieval index at {self.persist_dir} is empty. Build it with:\n"
                "  python -m adapter.cli build-index"
            )

    def search(
        self,
        query: str,
        k: int = 5,
        collections: Iterable[str] | None = None,
    ) -> list[SearchHit]:
        """Return the *k* best passages across the requested collections.

        Results are merged across collections and re-ranked by score, because a
        caller asking "fix nvt" wants the best answer wherever it lives — not
        five results from whichever collection happened to be queried first.

        Raises:
            ValueError: the query is empty or *k* is not positive.
            IndexNotBuiltError: nothing has been indexed yet.
        """
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if k <= 0:
            raise ValueError("k must be positive")

        self.require_built()
        wanted = list(collections) if collections is not None else list(COLLECTIONS)
        unknown = [c for c in wanted if c not in COLLECTIONS]
        if unknown:
            raise ValueError(f"unknown collections {unknown}; expected {COLLECTIONS}")

        client = self._require_client()
        available = {c.name for c in client.list_collections()}

        hits: list[SearchHit] = []
        for name in wanted:
            if name not in available:
                continue
            collection = self._collection(name)
            if collection.count() == 0:
                continue
            response = collection.query(
                query_texts=[query],
                n_results=min(k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            hits.extend(_to_hits(name, response))

        # Cosine distance in [0, 2]; convert to a similarity so higher is better.
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


def _jsonable(metadata: dict[str, Any]) -> dict[str, Any]:
    """ChromaDB metadata values must be scalar; drop or stringify the rest."""
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif value is None:
            continue
        else:
            out[key] = str(value)
    return out


def _to_hits(name: str, response: dict[str, Any]) -> list[SearchHit]:
    """Flatten one ChromaDB query response into hits."""
    documents = (response.get("documents") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]
    ids = (response.get("ids") or [[]])[0]

    hits: list[SearchHit] = []
    for index, text in enumerate(documents):
        metadata = dict(metadatas[index]) if index < len(metadatas) and metadatas[index] else {}
        distance = float(distances[index]) if index < len(distances) else 1.0
        hits.append(
            SearchHit(
                doc_id=ids[index] if index < len(ids) else f"{name}:{index}",
                collection=name,
                text=text,
                metadata=metadata,
                score=1.0 - distance,
            )
        )
    return hits
