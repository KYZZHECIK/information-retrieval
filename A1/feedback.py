"""Pseudo-relevance feedback (Rocchio-style)."""

from collections import Counter, defaultdict

from A1.index import InvertedIndex


def rocchio_expand(
    query_terms: dict[str, float],
    top_doc_ids: list[int],
    index: InvertedIndex,
    alpha: float = 1.0,
    beta: float = 0.3,
    num_expand_terms: int = 20,
) -> dict[str, float]:
    """Expand query using Rocchio-style pseudo-relevance feedback.

    Takes top-k retrieved document internal IDs, extracts discriminative terms
    weighted by IDF * tf_in_feedback_docs, and adds them to the query.

    Args:
        query_terms: original query term weights
        top_doc_ids: internal IDs of top-k pseudo-relevant documents
        index: the inverted index
        alpha: weight for original query terms
        beta: weight for expansion terms
        num_expand_terms: number of terms to add

    Returns:
        Expanded query term weights
    """
    if not top_doc_ids:
        return query_terms

    # Collect term frequencies across feedback documents
    feedback_tf: dict[str, float] = defaultdict(float)
    for term, postings in index.postings.items():
        idf = index.idf.get(term, 0.0)
        if idf <= 0:
            continue
        for doc_id, tf in postings:
            if doc_id in _fast_set(top_doc_ids):
                feedback_tf[term] += tf * idf

    # Normalize by number of feedback docs
    n_fb = len(top_doc_ids)
    for term in feedback_tf:
        feedback_tf[term] /= n_fb

    # Select top expansion terms (excluding original query terms)
    original_terms = set(query_terms.keys())
    candidates = [
        (term, score)
        for term, score in feedback_tf.items()
        if term not in original_terms
    ]
    candidates.sort(key=lambda x: -x[1])
    expand_terms = candidates[:num_expand_terms]

    # Build expanded query
    expanded = {}
    for term, weight in query_terms.items():
        expanded[term] = alpha * weight

    if expand_terms:
        max_expand_score = expand_terms[0][1] if expand_terms else 1.0
        for term, score in expand_terms:
            # Normalize expansion term weights relative to the strongest one
            expanded[term] = beta * (score / max_expand_score)

    return expanded


def _fast_set(doc_ids: list[int]) -> frozenset[int]:
    """Convert doc_ids to a frozenset for O(1) membership checks."""
    return frozenset(doc_ids)
