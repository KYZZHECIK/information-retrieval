"""CLI entry point for the IR retrieval system."""

import argparse
import sys
from collections import Counter
from pathlib import Path

from A1.feedback import rocchio_expand
from A1.index import InvertedIndex, build_index
from A1.parse import parse_documents, parse_topics
from A1.preprocessing import build_stopset, case_fold, stem, tokenize
from A1.score import score_bm25plus, score_cosine

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data" / "A1")


# ── Preset configurations ──────────────────────────────────────────────

PRESETS = {
    "run-0": dict(
        scorer="cosine",
        lowercase=False,
        do_stem=False,
        stopwords_df=0.0,
        extract_fields=False,
        bm25_k1=1.2,
        bm25_b=0.75,
        bm25_delta=1.0,
        title_boost=0.0,
        prf_k=0,
        prf_m=0,
        prf_beta=0.0,
        use_desc=False,
        use_narr=False,
    ),
    "run-1": dict(
        scorer="bm25plus",
        lowercase=True,
        do_stem=True,
        stopwords_df=0.4,
        extract_fields=True,
        bm25_k1=1.2,
        bm25_b=0.75,
        bm25_delta=1.0,
        title_boost=3.0,
        prf_k=10,
        prf_m=20,
        prf_beta=0.3,
        use_desc=False,
        use_narr=False,
    ),
    "run-2": dict(
        scorer="bm25plus",
        lowercase=True,
        do_stem=True,
        stopwords_df=0.4,
        extract_fields=True,
        bm25_k1=1.2,
        bm25_b=0.75,
        bm25_delta=1.0,
        title_boost=3.0,
        prf_k=15,
        prf_m=30,
        prf_beta=0.4,
        use_desc=True,
        use_narr=True,
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NPFL103 Vector Space IR System")
    p.add_argument("-q", "--topics", required=True, help="Topics XML file")
    p.add_argument("-d", "--docs", required=True, help="Document list file")
    p.add_argument("-r", "--run-id", required=True, help="Run identifier")
    p.add_argument("-o", "--output", required=True, help="Output result file")
    p.add_argument("--lang", choices=["en", "cs"], default=None,
                   help="Language (auto-detected from topics filename if omitted)")
    p.add_argument("--preset", choices=["run-0", "run-1", "run-2"], default=None)
    p.add_argument("--data-dir", default=DATA_DIR)

    # Scoring options
    p.add_argument("--scorer", choices=["cosine", "bm25plus"], default=None)
    p.add_argument("--lowercase", action="store_true", default=None)
    p.add_argument("--no-lowercase", action="store_false", dest="lowercase")
    p.add_argument("--stem", action="store_true", default=None, dest="do_stem")
    p.add_argument("--no-stem", action="store_false", dest="do_stem")
    p.add_argument("--stopwords-df", type=float, default=None)
    p.add_argument("--extract-fields", action="store_true", default=None)
    p.add_argument("--no-extract-fields", action="store_false", dest="extract_fields")

    # BM25+ parameters
    p.add_argument("--bm25-k1", type=float, default=None)
    p.add_argument("--bm25-b", type=float, default=None)
    p.add_argument("--bm25-delta", type=float, default=None)
    p.add_argument("--title-boost", type=float, default=None)

    # PRF parameters
    p.add_argument("--prf-k", type=int, default=None)
    p.add_argument("--prf-m", type=int, default=None)
    p.add_argument("--prf-beta", type=float, default=None)

    # Query construction
    p.add_argument("--use-desc", action="store_true", default=None)
    p.add_argument("--no-use-desc", action="store_false", dest="use_desc")
    p.add_argument("--use-narr", action="store_true", default=None)
    p.add_argument("--no-use-narr", action="store_false", dest="use_narr")

    args = p.parse_args(argv)

    # Auto-detect language from topics filename
    if args.lang is None:
        tname = Path(args.topics).name
        if "_cs" in tname:
            args.lang = "cs"
        elif "_en" in tname:
            args.lang = "en"
        else:
            p.error("Cannot auto-detect language; use --lang")

    # Apply preset, then override with explicit CLI args
    if args.preset:
        preset = PRESETS[args.preset]
        for key, default_val in preset.items():
            if getattr(args, key) is None:
                setattr(args, key, default_val)

    # Fill remaining None values with run-0 defaults
    defaults = PRESETS["run-0"]
    for key, default_val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default_val)

    return args


def build_normalize_fn(args):
    """Build a normalization function chain from args."""
    steps = []
    if args.lowercase:
        steps.append(lambda tokens: case_fold(tokens))
    if args.do_stem:
        lang = args.lang
        steps.append(lambda tokens, l=lang: stem(tokens, l))

    if not steps:
        return None

    def normalize(tokens):
        for step in steps:
            tokens = step(tokens)
        return tokens

    return normalize


def build_query(topic, args, normalize_fn) -> dict[str, float]:
    """Construct query term weights from topic fields."""
    parts = [topic.title]

    if args.use_desc:
        parts.append(topic.desc)
    if args.use_narr:
        # Extract positive-signal text from narrative, avoiding negation phrases
        narr = topic.narr
        # Simple heuristic: split on sentences, keep those without negation markers
        sentences = narr.replace(". ", ".\n").split("\n")
        positive = []
        neg_markers = {"not relevant", "not of interest", "are not", "is not",
                       "nejsou relevantní", "nejsou předmětem", "nejsou podstatná"}
        for sent in sentences:
            sent_lower = sent.lower()
            if not any(m in sent_lower for m in neg_markers):
                positive.append(sent)
        if positive:
            parts.append(" ".join(positive))

    text = " ".join(parts)
    tokens = tokenize(text)
    if normalize_fn:
        tokens = normalize_fn(tokens)

    # Weight: title terms get weight 1.0, desc terms 0.5, narr terms 0.3
    # For simplicity with mixed parts, use uniform tf-based weights
    tf = Counter(tokens)
    return {term: float(count) for term, count in tf.items()}


def write_results(
    results: list[tuple[str, float]], qid: str, run_id: str, fout
):
    """Write results for one query in TREC format."""
    for rank, (docno, score) in enumerate(results[:1000]):
        fout.write(f"{qid} 0 {docno} {rank} {score:.6f} {run_id}\n")


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    print(f"[{args.run_id}] Loading topics from {args.topics}", file=sys.stderr)
    topics = parse_topics(args.topics)
    print(f"[{args.run_id}] Loaded {len(topics)} topics", file=sys.stderr)

    # Build normalization function
    normalize_fn = build_normalize_fn(args)

    # Parse and index documents
    print(f"[{args.run_id}] Indexing documents ({args.lang})...", file=sys.stderr)
    docs = parse_documents(
        args.docs, args.data_dir, args.lang,
        extract_fields=args.extract_fields,
    )
    idx = build_index(docs, tokenize, normalize_fn, extract_fields=args.extract_fields)
    print(
        f"[{args.run_id}] Indexed {idx.num_docs} docs, "
        f"{len(idx.postings)} terms, avgdl={idx.avg_dl:.1f}",
        file=sys.stderr,
    )

    # Build stopword set if requested
    stopset: frozenset[str] = frozenset()
    if args.stopwords_df > 0:
        stopset = build_stopset(idx.df, idx.num_docs, args.stopwords_df)
        print(f"[{args.run_id}] Stopwords: {len(stopset)} terms (df > {args.stopwords_df})", file=sys.stderr)

    # Score and write results
    print(f"[{args.run_id}] Scoring with {args.scorer}...", file=sys.stderr)
    with open(args.output, "w") as fout:
        for i, topic in enumerate(topics):
            query = build_query(topic, args, normalize_fn)

            # Remove stopwords from query
            if stopset:
                query = {t: w for t, w in query.items() if t not in stopset}

            if not query:
                continue

            # Initial retrieval
            if args.scorer == "cosine":
                results = score_cosine(query, idx, top_k=1000)
            else:
                results = score_bm25plus(
                    query, idx,
                    top_k=1000,
                    k1=args.bm25_k1,
                    b=args.bm25_b,
                    delta=args.bm25_delta,
                    title_boost=args.title_boost,
                )

            # Pseudo-relevance feedback
            if args.prf_k > 0 and results:
                top_doc_ids = [
                    idx.docno_to_id[docno]
                    for docno, _ in results[:args.prf_k]
                ]
                expanded_query = rocchio_expand(
                    query, top_doc_ids, idx,
                    alpha=1.0,
                    beta=args.prf_beta,
                    num_expand_terms=args.prf_m,
                )
                # Remove stopwords from expanded query too
                if stopset:
                    expanded_query = {t: w for t, w in expanded_query.items() if t not in stopset}

                # Re-score with expanded query
                if args.scorer == "cosine":
                    results = score_cosine(expanded_query, idx, top_k=1000)
                else:
                    results = score_bm25plus(
                        expanded_query, idx,
                        top_k=1000,
                        k1=args.bm25_k1,
                        b=args.bm25_b,
                        delta=args.bm25_delta,
                        title_boost=args.title_boost,
                    )

            write_results(results, topic.qid, args.run_id, fout)

            if (i + 1) % 5 == 0:
                print(f"[{args.run_id}] Processed {i+1}/{len(topics)} topics", file=sys.stderr)

    print(f"[{args.run_id}] Results written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
