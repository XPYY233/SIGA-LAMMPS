"""R — retrieval over a LAMMPS knowledge base.

Three ChromaDB collections (examples, docs, syntax) built from a LAMMPS source
checkout. `corpora.py` discovers and chunks; `index.py` embeds and queries.
"""

from adapter.retrieval.corpora import (
    COLLECTIONS,
    Document,
    chunk_text,
    iter_documents,
    iter_docs,
    iter_examples,
    iter_syntax,
    parse_rst_page,
    strip_rst,
)

__all__ = [
    "COLLECTIONS",
    "Document",
    "chunk_text",
    "iter_documents",
    "iter_docs",
    "iter_examples",
    "iter_syntax",
    "parse_rst_page",
    "strip_rst",
]
