"""Grid search parameter optimization for run-1 and run-2."""

import csv
import itertools
import sys
import tempfile
from pathlib import Path

from A1.__main__ import main as run_main
from A1.evaluate import run_trec_eval

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data" / "A1")


def evaluate_config(
    lang: str,
    topics_file: str,
    docs_file: str,
    qrels_file: str,
    run_id: str,
    extra_args: list[str],
) -> dict[str, float]:
    """Run a single configuration and return trec_eval metrics."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".res", delete=False) as f:
        out_path = f.name

    argv = [
        "-q", topics_file,
        "-d", docs_file,
        "-r", run_id,
        "-o", out_path,
        "--data-dir", DATA_DIR,
        "--lang", lang,
    ] + extra_args

    run_main(argv)
    metrics = run_trec_eval(qrels_file, out_path)
    Path(out_path).unlink(missing_ok=True)
    return metrics


def sweep_single_param(
    lang: str,
    param_name: str,
    values: list,
    base_args: list[str],
    topics_file: str,
    docs_file: str,
    qrels_file: str,
):
    """Sweep one parameter while keeping others at base values."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Sweeping {param_name} for {lang.upper()}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    results = []
    for val in values:
        run_id = f"tune_{lang}_{param_name}_{val}"
        args = base_args + [f"--{param_name}", str(val)]
        try:
            metrics = evaluate_config(lang, topics_file, docs_file, qrels_file, run_id, args)
            m = metrics.get("map", 0)
            p10 = metrics.get("P_10", 0)
            results.append((val, m, p10))
            print(f"  {param_name}={val:>8} -> MAP={m:.4f}  P@10={p10:.4f}", file=sys.stderr)
        except Exception as e:
            print(f"  {param_name}={val:>8} -> ERROR: {e}", file=sys.stderr)

    results.sort(key=lambda x: -x[1])
    if results:
        best = results[0]
        print(f"  BEST: {param_name}={best[0]} MAP={best[1]:.4f}", file=sys.stderr)
    return results


def tune_language(lang: str):
    """Run parameter tuning for a single language."""
    topics_file = f"data/A1/topics-train_{lang}.xml"
    docs_file = f"documents_{lang}.lst"
    qrels_file = f"data/A1/qrels-train_{lang}.txt"

    # Base configuration (run-1 defaults)
    base_args = [
        "--scorer", "bm25plus",
        "--lowercase",
        "--stem",
        "--extract-fields",
    ]

    print(f"\n{'#'*60}", file=sys.stderr)
    print(f"# TUNING {lang.upper()}", file=sys.stderr)
    print(f"{'#'*60}", file=sys.stderr)

    # Phase 1: Sweep individual parameters
    # BM25+ k1
    sweep_single_param(
        lang, "bm25-k1", [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
        base_args + ["--bm25-b", "0.75", "--bm25-delta", "1.0",
                      "--stopwords-df", "0.4", "--title-boost", "3.0",
                      "--prf-k", "0"],
        topics_file, docs_file, qrels_file,
    )

    # BM25+ b
    sweep_single_param(
        lang, "bm25-b", [0.3, 0.5, 0.65, 0.75, 0.85],
        base_args + ["--bm25-k1", "1.2", "--bm25-delta", "1.0",
                      "--stopwords-df", "0.4", "--title-boost", "3.0",
                      "--prf-k", "0"],
        topics_file, docs_file, qrels_file,
    )

    # BM25+ delta
    sweep_single_param(
        lang, "bm25-delta", [0.0, 0.5, 1.0, 1.5, 2.0],
        base_args + ["--bm25-k1", "1.2", "--bm25-b", "0.75",
                      "--stopwords-df", "0.4", "--title-boost", "3.0",
                      "--prf-k", "0"],
        topics_file, docs_file, qrels_file,
    )

    # Title boost
    sweep_single_param(
        lang, "title-boost", [0.0, 1.0, 2.0, 3.0, 5.0, 8.0],
        base_args + ["--bm25-k1", "1.2", "--bm25-b", "0.75", "--bm25-delta", "1.0",
                      "--stopwords-df", "0.4", "--prf-k", "0"],
        topics_file, docs_file, qrels_file,
    )

    # Stopwords df threshold
    sweep_single_param(
        lang, "stopwords-df", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        base_args + ["--bm25-k1", "1.2", "--bm25-b", "0.75", "--bm25-delta", "1.0",
                      "--title-boost", "3.0", "--prf-k", "0"],
        topics_file, docs_file, qrels_file,
    )

    # PRF (combined: k, m, beta)
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Sweeping PRF for {lang.upper()}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    prf_configs = [
        ("off", ["--prf-k", "0"]),
        ("5_10_0.2", ["--prf-k", "5", "--prf-m", "10", "--prf-beta", "0.2"]),
        ("10_15_0.2", ["--prf-k", "10", "--prf-m", "15", "--prf-beta", "0.2"]),
        ("10_20_0.3", ["--prf-k", "10", "--prf-m", "20", "--prf-beta", "0.3"]),
        ("10_20_0.5", ["--prf-k", "10", "--prf-m", "20", "--prf-beta", "0.5"]),
        ("15_30_0.3", ["--prf-k", "15", "--prf-m", "30", "--prf-beta", "0.3"]),
        ("15_30_0.4", ["--prf-k", "15", "--prf-m", "30", "--prf-beta", "0.4"]),
    ]
    for name, prf_args in prf_configs:
        run_id = f"tune_{lang}_prf_{name}"
        args = base_args + [
            "--bm25-k1", "1.2", "--bm25-b", "0.75", "--bm25-delta", "1.0",
            "--stopwords-df", "0.4", "--title-boost", "3.0",
        ] + prf_args
        try:
            metrics = evaluate_config(lang, topics_file, docs_file, qrels_file, run_id, args)
            m = metrics.get("map", 0)
            p10 = metrics.get("P_10", 0)
            print(f"  PRF={name:>12} -> MAP={m:.4f}  P@10={p10:.4f}", file=sys.stderr)
        except Exception as e:
            print(f"  PRF={name:>12} -> ERROR: {e}", file=sys.stderr)


if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    tune_language(lang)
