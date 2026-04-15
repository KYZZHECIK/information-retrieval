"""Fast parameter tuning - builds index once, varies only scoring params."""

import sys
import tempfile
from collections import Counter
from pathlib import Path

from A1.evaluate import run_trec_eval
from A1.feedback import rocchio_expand
from A1.index import build_index
from A1.parse import parse_documents, parse_topics
from A1.preprocessing import build_stopset, case_fold, stem, tokenize
from A1.score import score_bm25plus

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data" / "A1")


def run_scoring(
    topics, idx, normalize_fn, stopset, lang, run_id,
    k1, b, delta, title_boost, prf_k, prf_m, prf_beta,
    qrels_path,
    proximity_boost=0.0, query_idf=False,
):
    """Score all topics with given params, evaluate, return metrics."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".res", delete=False) as f:
        out_path = f.name

    with open(out_path, "w") as fout:
        for topic in topics:
            tokens = tokenize(topic.title)
            if normalize_fn:
                tokens = normalize_fn(tokens)
            query = {t: float(c) for t, c in Counter(tokens).items()}
            if stopset:
                query = {t: w for t, w in query.items() if t not in stopset}
            if not query:
                continue

            results = score_bm25plus(
                query, idx, top_k=1000,
                k1=k1, b=b, delta=delta, title_boost=title_boost,
                proximity_boost=proximity_boost, query_idf=query_idf,
            )

            if prf_k > 0 and results:
                top_ids = [idx.docno_to_id[d] for d, _ in results[:prf_k]]
                eq = rocchio_expand(query, top_ids, idx, beta=prf_beta, num_expand_terms=prf_m)
                if stopset:
                    eq = {t: w for t, w in eq.items() if t not in stopset}
                results = score_bm25plus(
                    eq, idx, top_k=1000,
                    k1=k1, b=b, delta=delta, title_boost=title_boost,
                    proximity_boost=proximity_boost, query_idf=query_idf,
                )

            for rank, (docno, score) in enumerate(results[:1000]):
                fout.write(f"{topic.qid} 0 {docno} {rank} {score:.6f} {run_id}\n")

    metrics = run_trec_eval(qrels_path, out_path)
    Path(out_path).unlink(missing_ok=True)
    return metrics


def tune(lang: str):
    topics_file = f"data/A1/topics-train_{lang}.xml"
    docs_lst = f"documents_{lang}.lst"
    qrels_path = f"data/A1/qrels-train_{lang}.txt"

    topics = parse_topics(topics_file)
    print(f"Building index for {lang.upper()} (with positions)...", file=sys.stderr)

    def normalize_fn(tokens):
        return stem(case_fold(tokens), lang)

    docs = parse_documents(docs_lst, DATA_DIR, lang, extract_fields=True)
    idx = build_index(docs, tokenize, normalize_fn,
                      extract_fields=True, store_forward=True, store_positions=True)
    print(f"Indexed {idx.num_docs} docs, {len(idx.postings)} terms", file=sys.stderr)

    stopsets = {}
    for thr in [0.0, 0.3, 0.4, 0.5]:
        stopsets[thr] = build_stopset(idx.df, idx.num_docs, thr) if thr > 0 else frozenset()

    # Best params from previous tuning
    if lang == "en":
        base = dict(k1=1.4, b=0.2, delta=1.0, tb=3.0, sw=0.4, prf_k=0, prf_m=0, prf_beta=0)
    else:
        base = dict(k1=1.4, b=0.6, delta=1.0, tb=3.0, sw=0.4, prf_k=5, prf_m=15, prf_beta=0.3)

    best_map = 0
    best_config = {}

    def test(label, prox=0.0, qi=False, **overrides):
        nonlocal best_map, best_config
        cfg = {**base, **overrides}
        m = run_scoring(
            topics, idx, normalize_fn, stopsets[cfg["sw"]], lang, label,
            cfg["k1"], cfg["b"], cfg["delta"], cfg["tb"],
            cfg["prf_k"], cfg["prf_m"], cfg["prf_beta"], qrels_path,
            proximity_boost=prox, query_idf=qi,
        )
        _map = m.get("map", 0)
        _p10 = m.get("P_10", 0)
        extras = f"prox={prox} qi={qi}" if prox > 0 or qi else ""
        print(f"  {label:>40} MAP={_map:.4f} P@10={_p10:.4f}  {extras}", file=sys.stderr)
        if _map > best_map:
            best_map = _map
            best_config = {**cfg, "prox": prox, "qi": qi}
        return _map

    print(f"\n=== Baseline (previous best for {lang.upper()}) ===", file=sys.stderr)
    test("baseline")

    print(f"\n=== IDF query weighting ===", file=sys.stderr)
    test("query_idf=True", qi=True)

    print(f"\n=== Proximity boost sweep ===", file=sys.stderr)
    for prox in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        test(f"prox={prox}", prox=prox)

    print(f"\n=== Proximity + IDF query weighting ===", file=sys.stderr)
    for prox in [0.0, 1.0, 2.0, 5.0]:
        test(f"prox={prox}+qi", prox=prox, qi=True)

    print(f"\n=== Proximity + adjusted params ===", file=sys.stderr)
    # Try proximity with different b and title_boost values
    for prox in [1.0, 3.0, 5.0]:
        for b_val in [0.1, 0.3, 0.5]:
            if lang == "en":
                test(f"prox={prox}_b={b_val}", prox=prox, b=b_val)
            else:
                test(f"prox={prox}_b={b_val}", prox=prox, b=b_val)

    # Try different title_boost with proximity
    print(f"\n=== Proximity + title_boost sweep ===", file=sys.stderr)
    bp = best_config.get("prox", 0)
    for tb in [0.0, 1.0, 2.0, 3.0, 5.0]:
        test(f"prox={bp}_tb={tb}", prox=bp, tb=tb)

    # Try combining everything
    print(f"\n=== Combined: best proximity + PRF ===", file=sys.stderr)
    bqi = best_config.get("qi", False)
    for pk, pm, pbeta in [(0, 0, 0), (5, 10, 0.2), (5, 15, 0.3), (10, 20, 0.3)]:
        test(f"combined_prf_{pk}_{pm}_{pbeta}", prox=bp, qi=bqi,
             prf_k=pk, prf_m=pm, prf_beta=pbeta)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"BEST {lang.upper()}: MAP={best_map:.4f} config={best_config}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    tune(lang)
