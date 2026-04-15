"""Inverted index construction and storage."""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from A1.parse import Document


@dataclass
class DocInfo:
    docno: str
    length: int = 0
    l2_norm: float = 0.0
    title_length: int = 0


@dataclass
class InvertedIndex:
    # term -> sorted list of (doc_internal_id, tf)
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    # title/heading field postings (same structure)
    title_postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    # positional index: term -> {doc_id: [pos0, pos1, ...]}
    # Only populated when store_positions=True
    positions: dict[str, dict[int, list[int]]] = field(default_factory=dict)
    # forward index: doc_id -> {term: tf} (only populated when store_forward=True)
    doc_terms: list[dict[str, int]] = field(default_factory=list)
    # document metadata indexed by internal id
    doc_meta: list[DocInfo] = field(default_factory=list)
    # docno string -> internal int id
    docno_to_id: dict[str, int] = field(default_factory=dict)
    # precomputed statistics
    num_docs: int = 0
    avg_dl: float = 0.0
    # document frequency: term -> number of docs containing it
    df: dict[str, int] = field(default_factory=dict)
    # precomputed IDF values
    idf: dict[str, float] = field(default_factory=dict)

    def compute_idf(self):
        """Compute IDF for all terms: log((N - df + 0.5) / (df + 0.5) + 1)."""
        n = self.num_docs
        for term, df_val in self.df.items():
            self.idf[term] = math.log((n - df_val + 0.5) / (df_val + 0.5) + 1.0)


def build_index(
    documents,
    tokenize_fn: Callable[[str], list[str]],
    normalize_fn: Callable[[list[str]], list[str]] | None = None,
    extract_fields: bool = False,
    store_forward: bool = False,
    store_positions: bool = False,
) -> InvertedIndex:
    """Build an inverted index from a document stream.

    Args:
        documents: iterable of Document objects
        tokenize_fn: tokenizer function
        normalize_fn: optional normalization pipeline (case fold + stem + stopword removal)
        extract_fields: if True, also build title_postings
        store_forward: if True, store forward index (doc_id -> {term: tf}) for fast PRF
        store_positions: if True, store term positions for proximity scoring
    """
    idx = InvertedIndex()
    total_length = 0

    for doc in documents:
        doc_id = len(idx.doc_meta)
        idx.docno_to_id[doc.docno] = doc_id

        # Tokenize and normalize full text
        tokens = tokenize_fn(doc.full_text)
        if normalize_fn:
            tokens = normalize_fn(tokens)

        tf_counts = Counter(tokens)
        doc_length = len(tokens)
        l2_norm = math.sqrt(sum(c * c for c in tf_counts.values())) if tf_counts else 0.0

        # Build position lists
        if store_positions:
            term_positions: dict[str, list[int]] = {}
            for pos, term in enumerate(tokens):
                if term not in term_positions:
                    term_positions[term] = []
                term_positions[term].append(pos)
            for term, positions in term_positions.items():
                if term not in idx.positions:
                    idx.positions[term] = {}
                idx.positions[term][doc_id] = positions

        # Title field processing
        title_length = 0
        title_tf: Counter[str] = Counter()
        if extract_fields and doc.title_text:
            title_tokens = tokenize_fn(doc.title_text)
            if normalize_fn:
                title_tokens = normalize_fn(title_tokens)
            title_tf = Counter(title_tokens)
            title_length = len(title_tokens)

        info = DocInfo(
            docno=doc.docno,
            length=doc_length,
            l2_norm=l2_norm,
            title_length=title_length,
        )
        idx.doc_meta.append(info)
        total_length += doc_length

        # Store forward index for PRF
        if store_forward:
            idx.doc_terms.append(dict(tf_counts))

        # Add to main postings
        for term, count in tf_counts.items():
            if term not in idx.postings:
                idx.postings[term] = []
            idx.postings[term].append((doc_id, count))

        # Add to title postings
        if extract_fields:
            for term, count in title_tf.items():
                if term not in idx.title_postings:
                    idx.title_postings[term] = []
                idx.title_postings[term].append((doc_id, count))

    idx.num_docs = len(idx.doc_meta)
    idx.avg_dl = total_length / idx.num_docs if idx.num_docs else 0.0

    # Compute document frequencies
    for term, plist in idx.postings.items():
        idx.df[term] = len(plist)

    # Compute IDF
    idx.compute_idf()

    return idx
