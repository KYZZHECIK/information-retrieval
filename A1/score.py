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


def _min_window_size(position_lists: list[list[int]]) -> int:
    """Find the minimum window containing at least one element from each list.

    Uses a pointer-based sweep algorithm. O(n log k) where n = total positions.
    Returns the window size, or a large number if impossible.
    """
    k = len(position_lists)
    if k <= 1:
        return 1

    # Initialize pointers: (position, list_index)
    import heapq as hq
    pointers = []
    max_pos = 0
    for i, plist in enumerate(position_lists):
        if not plist:
            return 10000  # term not in this doc
        pointers.append((plist[0], i, 0))  # (pos, list_idx, offset_in_list)
        max_pos = max(max_pos, plist[0])
    hq.heapify(pointers)

    best = max_pos - pointers[0][0] + 1

    while True:
        min_pos, list_idx, offset = hq.heappop(pointers)
        offset += 1
        if offset >= len(position_lists[list_idx]):
            break
        new_pos = position_lists[list_idx][offset]
        max_pos = max(max_pos, new_pos)
        hq.heappush(pointers, (new_pos, list_idx, offset))
        window = max_pos - pointers[0][0] + 1
        if window < best:
            best = window

    return best


def score_bm25plus(
    query_terms: dict[str, float],
    index: InvertedIndex,
    top_k: int = 1000,
    k1: float = 1.2,
    b: float = 0.75,
    delta: float = 1.0,
    title_boost: float = 0.0,
    proximity_boost: float = 0.0,
    query_idf: bool = False,
) -> list[tuple[str, float]]:
    """BM25+ scorer with optional proximity boost and IDF query weighting.

    score(t,d) = IDF(t) * [(k1+1)*tf_eff / (k1*(1-b+b*dl/avgdl) + tf_eff) + delta]

    Optional enhancements:
    - proximity_boost > 0: adds a bonus for docs where query terms appear close together
    - query_idf: multiply query term weights by IDF for discriminative weighting
    """
    accumulators: dict[int, float] = defaultdict(float)
    # Track which query terms match each doc (for proximity + coordination)
    doc_matched_terms: dict[int, int] = defaultdict(int) if proximity_boost > 0 else {}
    avg_dl = index.avg_dl

    # Precompute title postings lookup
    title_lookup: dict[str, dict[int, int]] = {}
    if title_boost > 0:
        for term in query_terms:
            if term in index.title_postings:
                title_lookup[term] = dict(index.title_postings[term])

    # Optionally apply IDF weighting to query terms
    effective_query = query_terms
    if query_idf:
        effective_query = {}
        for term, w in query_terms.items():
            idf = index.idf.get(term, 0.0)
            effective_query[term] = w * idf if idf > 0 else w

    for term, q_weight in effective_query.items():
        if term not in index.postings:
            continue

        idf = index.idf.get(term, 0.0)
        title_map = title_lookup.get(term, {})

        for doc_id, tf_raw in index.postings[term]:
            tf_eff = tf_raw
            if title_boost > 0 and doc_id in title_map:
                tf_eff += title_boost * title_map[doc_id]

            dl = index.doc_meta[doc_id].length
            denom = k1 * (1.0 - b + b * dl / avg_dl) + tf_eff
            tf_component = (k1 + 1.0) * tf_eff / denom + delta

            accumulators[doc_id] += idf * tf_component * q_weight

            if proximity_boost > 0:
                doc_matched_terms[doc_id] += 1

    # Apply proximity boost for docs matching 2+ query terms
    if proximity_boost > 0 and index.positions:
        query_term_list = [t for t in effective_query if t in index.postings]
        n_query_terms = len(query_term_list)

        if n_query_terms >= 2:
            for doc_id in list(accumulators.keys()):
                if doc_matched_terms.get(doc_id, 0) < 2:
                    continue
                # Collect position lists for query terms in this doc
                pos_lists = []
                for term in query_term_list:
                    if term in index.positions and doc_id in index.positions[term]:
                        pos_lists.append(index.positions[term][doc_id])
                if len(pos_lists) >= 2:
                    window = _min_window_size(pos_lists)
                    # Proximity bonus: inversely proportional to window size
                    # Normalized by number of query terms
                    bonus = proximity_boost * (len(pos_lists) / window)
                    accumulators[doc_id] += bonus

    results = list(accumulators.items())
    if len(results) <= top_k:
        results.sort(key=lambda x: (-x[1], x[0]))
    else:
        results = heapq.nsmallest(top_k, results, key=lambda x: (-x[1], x[0]))

    return [
        (index.doc_meta[doc_id].docno, score) for doc_id, score in results
    ]
