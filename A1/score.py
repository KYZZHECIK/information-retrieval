"""Scoring functions for document retrieval."""

import heapq
import math
from collections import defaultdict

from A1.index import InvertedIndex


def score_cosine(
    query_terms: dict[str, float],
    index: InvertedIndex,
    top_k: int = 1000,
) -> list[tuple[str, float]]:
    """Cosine similarity scorer for run-0 baseline.

    Uses natural tf (raw count) for both query and document vectors.
    No IDF weighting. Cosine normalization on both sides.
    """
    accumulators: dict[int, float] = defaultdict(float)

    for term, q_weight in query_terms.items():
        if term not in index.postings:
            continue
        for doc_id, tf_doc in index.postings[term]:
            accumulators[doc_id] += tf_doc * q_weight

    # Query L2 norm
    q_norm = math.sqrt(sum(w * w for w in query_terms.values()))
    if q_norm == 0:
        return []

    # Normalize by document L2 norm and query L2 norm
    results = []
    for doc_id, raw_score in accumulators.items():
        d_norm = index.doc_meta[doc_id].l2_norm
        if d_norm == 0:
            continue
        score = raw_score / (d_norm * q_norm)
        results.append((doc_id, score))

    # Get top-k by score
    if len(results) <= top_k:
        results.sort(key=lambda x: (-x[1], x[0]))
    else:
        results = heapq.nsmallest(top_k, results, key=lambda x: (-x[1], x[0]))

    return [
        (index.doc_meta[doc_id].docno, score) for doc_id, score in results
    ]


def score_bm25plus(
    query_terms: dict[str, float],
    index: InvertedIndex,
    top_k: int = 1000,
    k1: float = 1.2,
    b: float = 0.75,
    delta: float = 1.0,
    title_boost: float = 0.0,
) -> list[tuple[str, float]]:
    """BM25+ scorer (Lv & Zhai, 2011).

    BM25+ adds a delta to the tf component to fix the lower-bounding issue
    where standard BM25 can underweight matching terms in long documents.

    score(t,d) = IDF(t) * [(k1+1)*tf_eff / (k1*(1-b+b*dl/avgdl) + tf_eff) + delta]

    With field weighting: tf_eff = tf_body + title_boost * tf_title
    """
    accumulators: dict[int, float] = defaultdict(float)
    avg_dl = index.avg_dl

    # Precompute title postings lookup for efficiency
    title_lookup: dict[str, dict[int, int]] = {}
    if title_boost > 0:
        for term in query_terms:
            if term in index.title_postings:
                title_lookup[term] = dict(index.title_postings[term])

    for term, q_weight in query_terms.items():
        if term not in index.postings:
            continue

        idf = index.idf.get(term, 0.0)
        title_map = title_lookup.get(term, {})

        for doc_id, tf_raw in index.postings[term]:
            # Field-weighted effective tf
            tf_eff = tf_raw
            if title_boost > 0 and doc_id in title_map:
                tf_eff += title_boost * title_map[doc_id]

            dl = index.doc_meta[doc_id].length
            denom = k1 * (1.0 - b + b * dl / avg_dl) + tf_eff
            tf_component = (k1 + 1.0) * tf_eff / denom + delta

            accumulators[doc_id] += idf * tf_component * q_weight

    results = list(accumulators.items())
    if len(results) <= top_k:
        results.sort(key=lambda x: (-x[1], x[0]))
    else:
        results = heapq.nsmallest(top_k, results, key=lambda x: (-x[1], x[0]))

    return [
        (index.doc_meta[doc_id].docno, score) for doc_id, score in results
    ]
