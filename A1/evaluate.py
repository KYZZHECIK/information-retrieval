"""Wrapper for trec_eval evaluation tool."""

import subprocess
from pathlib import Path

TREC_EVAL_BIN = str(
    Path(__file__).resolve().parent.parent / "data" / "A1" / "trec_eval-9.0.7" / "trec_eval"
)


def run_trec_eval(
    qrels_path: str,
    results_path: str,
    trec_eval_bin: str = TREC_EVAL_BIN,
) -> dict[str, float]:
    """Run trec_eval and parse output into a dict.

    Returns dict with keys like 'map', 'P_10', 'iprec_at_recall_0.00', etc.
    """
    result = subprocess.run(
        [trec_eval_bin, "-M1000", qrels_path, results_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"trec_eval failed: {result.stderr}")

    metrics = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 3:
            key = parts[0]
            try:
                value = float(parts[2])
            except ValueError:
                value = parts[2]
            metrics[key] = value
    return metrics
