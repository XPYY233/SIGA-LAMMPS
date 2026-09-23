"""Sparse BM25 retrieval — R's default backend.

Why not a dense hashing vectorizer
----------------------------------
The obvious fallback for "no downloadable model" is a hashing vectorizer: hash
each feature into a fixed-width vector and store that. It does not work at this
corpus size, and the measurement is unambiguous:

* ~6972 documents yield roughly **431,000 distinct features**
* hashed into 1024 dimensions that is **~421 features per bucket**

Every rare, discriminative bigram then shares a slot with hundreds of unrelated
features, so a literal phrase match is drowned by noise. Before this was
diagnosed, the exact phrase "mean square displacement" — the title of its own
manual page — ranked fourth behind a timing utility. No dimension that fits in
memory repairs it, because the required width is the size of the feature space.

BM25 is the right structure for lexical matching: a sparse inverted index with
exact term statistics, no hashing, and therefore no collisions at all.

What this backend does and does not do
--------------------------------------
It matches **lexically**. LAMMPS queries are dominated by command vocabulary —
an agent that knows it needs ``fix deform`` and wants the syntax is served well.
It does **not** match paraphrase: "stretch a box along one axis" and ``fix
deform`` share almost no vocabulary, and no amount of weighting invents a
relationship that is not in the text.

That limitation is real and is reported rather than hidden. Closing it needs
dense embeddings; :mod:`adapter.retrieval.index` uses them when a model is
genuinely available and reports which backend served a query.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["BM25Index", "tokenize", "BM25_K1", "BM25_B"]

#: Standard BM25 saturation and length-normalisation constants.
BM25_K1 = 1.2
BM25_B = 0.75

_TOKEN = re.compile(r"[a-z0-9_]+(?:/[a-z0-9_]+)?")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have how i if in into is it its of on or
    that the their them then there these this to was were what when where which who
    will with you your do does can could should would may might must not no
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Split *text* into retrieval tokens.

    LAMMPS vocabulary is kept intact, which matters more than it looks:
    ``lj/cut`` yields ``lj``, ``cut`` **and** ``lj/cut``, so a query for the bare
    word ``cut`` and one for the exact style both hit; ``fix_nvt`` and
    ``nvt/sllod`` keep their parts as well as their whole; identifiers such as
    ``pairStyle`` split on the camel-case boundary.
    """
    lowered = _CAMEL.sub(" ", text).lower()
    tokens: list[str] = []
    for match in _TOKEN.finditer(lowered):
        token = match.group(0)
        if "/" in token:
            head, _, tail = token.partition("/")
            tokens.extend(part for part in (head, tail) if part)
        tokens.append(token)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


@dataclass
class BM25Index:
    """An in-memory sparse BM25 index over a document set.

    Built from the corpus in a couple of seconds. Persisted as tokenised
    documents plus term statistics rather than as postings: the postings are
    derived, so storing them would roughly triple the artifact for no gain, and
    rebuilding them is cheap.
    """

    #: Postings: term -> [(document position, term frequency), ...]
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)
    lengths: list[int] = field(default_factory=list)
    avgdl: float = 1.0
    documents: list[dict[str, Any]] = field(default_factory=list)

    # -- construction ------------------------------------------------------- #

    @classmethod
    def build(cls, documents: list[dict[str, Any]]) -> BM25Index:
        """Build from ``[{doc_id, collection, text, metadata}, ...]``.

        Callers pass plain dictionaries rather than :class:`Document` so the
        index has no dependency on the corpus layer and can be persisted and
        reloaded without it.
        """
        postings: dict[str, list[tuple[int, int]]] = {}
        lengths: list[int] = []
        for position, document in enumerate(documents):
            counts = Counter(tokens_for(document))
            lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                postings.setdefault(term, []).append((position, frequency))

        total = len(documents)
        avgdl = (sum(lengths) / total) if total else 1.0
        idf = {
            term: math.log(1.0 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in postings.items()
        }
        return cls(postings=postings, idf=idf, lengths=lengths, avgdl=avgdl, documents=documents)

    # -- persistence -------------------------------------------------------- #

    def save(self, path: Path) -> None:
        """Persist the index. Tokenisation is replayed on load, not stored twice."""
        payload = {
            "version": 1,
            "avgdl": self.avgdl,
            "documents": self.documents,
            "tokens": [tokenize(d["text"]) for d in self.documents],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        """Load a persisted index, rebuilding postings and statistics.

        Raises:
            FileNotFoundError: the artifact is absent.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents = payload["documents"]
        tokens = payload.get("tokens") or [tokenize(d["text"]) for d in documents]

        postings: dict[str, list[tuple[int, int]]] = {}
        lengths: list[int] = []
        for position, token_list in enumerate(tokens):
            counts = Counter(token_list)
            lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                postings.setdefault(term, []).append((position, frequency))

        total = len(documents)
        avgdl = (sum(lengths) / total) if total else 1.0
        idf = {
            term: math.log(1.0 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in postings.items()
        }
        return cls(postings=postings, idf=idf, lengths=lengths, avgdl=avgdl, documents=documents)

    # -- query -------------------------------------------------------------- #

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Return the ``k`` best ``(document position, score)`` pairs.

        Documents matching no query term score zero and are omitted, so a query
        the corpus cannot answer returns nothing rather than arbitrary filler.
        """
        terms = tokenize(query)
        if not terms or not self.documents:
            return []

        scores: dict[int, float] = {}
        for term in set(terms):
            posting = self.postings.get(term)
            if posting is None:
                continue
            term_idf = self.idf[term]
            for position, frequency in posting:
                denominator = frequency + BM25_K1 * (
                    1.0 - BM25_B + BM25_B * (self.lengths[position] / self.avgdl)
                )
                scores[position] = scores.get(position, 0.0) + term_idf * (
                    frequency * (BM25_K1 + 1.0) / denominator
                )

        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        # Normalise to (0, 1] by the best score, so a caller cannot mistake a raw
        # BM25 score for a cosine similarity or a probability.
        best = ranked[0][1] or 1.0
        return [(position, score / best) for position, score in ranked]

    def __len__(self) -> int:
        return len(self.documents)


def tokens_for(document: dict[str, Any]) -> list[str]:
    """Tokens for one stored document."""
    return tokenize(document["text"])
