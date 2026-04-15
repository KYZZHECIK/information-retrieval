#!/usr/bin/env bash
set -e

EVAL="data/A1/trec_eval-9.0.7/trec_eval"
TOTAL=12
COUNT=0
START=$(date +%s)

progress() {
  COUNT=$((COUNT + 1))
  NOW=$(date +%s)
  ELAPSED=$(( NOW - START ))
  MINS=$(( ELAPSED / 60 ))
  SECS=$(( ELAPSED % 60 ))
  echo ""
  echo "✓ [$COUNT/$TOTAL] $1  (elapsed: ${MINS}m${SECS}s)"
  echo ""
}

eval_train() {
  local res="$1"
  local qrels="$2"
  if [ -f "$res" ]; then
    echo "  → $($EVAL -M1000 "$qrels" "$res" | grep -E '^(map|P_10)' | awk '{printf "%s=%s  ", $1, $3}')"
  fi
}

echo "============================================"
echo "  Generating all 12 result files"
echo "  Started at $(date '+%H:%M:%S')"
echo "============================================"

# ── Run-0 (baseline) ──────────────────────────
echo ""
echo "── Run-0: Baseline ──"

./run -q data/A1/topics-train_en.xml -d documents_en.lst -r run-0_en -o run-0_train_en.res --preset run-0
progress "run-0_train_en.res"
eval_train run-0_train_en.res data/A1/qrels-train_en.txt

./run -q data/A1/topics-test_en.xml  -d documents_en.lst -r run-0_en -o run-0_test_en.res  --preset run-0
progress "run-0_test_en.res"

./run -q data/A1/topics-train_cs.xml -d documents_cs.lst -r run-0_cs -o run-0_train_cs.res --preset run-0
progress "run-0_train_cs.res"
eval_train run-0_train_cs.res data/A1/qrels-train_cs.txt

./run -q data/A1/topics-test_cs.xml  -d documents_cs.lst -r run-0_cs -o run-0_test_cs.res  --preset run-0
progress "run-0_test_cs.res"

# ── Run-1 (tuned BM25+) ──────────────────────
echo ""
echo "── Run-1: Tuned BM25+ ──"

./run -q data/A1/topics-train_en.xml -d documents_en.lst -r run-1_en -o run-1_train_en.res --preset run-1
progress "run-1_train_en.res"
eval_train run-1_train_en.res data/A1/qrels-train_en.txt

./run -q data/A1/topics-test_en.xml  -d documents_en.lst -r run-1_en -o run-1_test_en.res  --preset run-1
progress "run-1_test_en.res"

./run -q data/A1/topics-train_cs.xml -d documents_cs.lst -r run-1_cs -o run-1_train_cs.res --preset run-1
progress "run-1_train_cs.res"
eval_train run-1_train_cs.res data/A1/qrels-train_cs.txt

./run -q data/A1/topics-test_cs.xml  -d documents_cs.lst -r run-1_cs -o run-1_test_cs.res  --preset run-1
progress "run-1_test_cs.res"

# ── Run-2 (unconstrained) ────────────────────
echo ""
echo "── Run-2: Unconstrained ──"

./run -q data/A1/topics-train_en.xml -d documents_en.lst -r run-2_en -o run-2_train_en.res --preset run-2
progress "run-2_train_en.res"
eval_train run-2_train_en.res data/A1/qrels-train_en.txt

./run -q data/A1/topics-test_en.xml  -d documents_en.lst -r run-2_en -o run-2_test_en.res  --preset run-2
progress "run-2_test_en.res"

./run -q data/A1/topics-train_cs.xml -d documents_cs.lst -r run-2_cs -o run-2_train_cs.res --preset run-2
progress "run-2_train_cs.res"
eval_train run-2_train_cs.res data/A1/qrels-train_cs.txt

./run -q data/A1/topics-test_cs.xml  -d documents_cs.lst -r run-2_cs -o run-2_test_cs.res  --preset run-2
progress "run-2_test_cs.res"

# ── Final summary ─────────────────────────────
echo ""
echo "============================================"
echo "  Final Evaluation Summary"
echo "============================================"
echo ""
printf "%-25s %8s %8s\n" "Result File" "MAP" "P@10"
printf "%-25s %8s %8s\n" "-------------------------" "--------" "--------"

for run in run-0 run-1 run-2; do
  for lang in en cs; do
    res="${run}_train_${lang}.res"
    qrels="data/A1/qrels-train_${lang}.txt"
    if [ -f "$res" ]; then
      map=$($EVAL -M1000 "$qrels" "$res" | grep "^map" | awk '{print $3}')
      p10=$($EVAL -M1000 "$qrels" "$res" | grep "^P_10" | awk '{print $3}')
      printf "%-25s %8s %8s\n" "$res" "$map" "$p10"
    fi
  done
done

NOW=$(date +%s)
ELAPSED=$(( NOW - START ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "============================================"
echo "  Done. 12 files in ${MINS}m${SECS}s"
echo "============================================"
ls -lh run-*.res
