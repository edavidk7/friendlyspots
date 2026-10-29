#!/usr/bin/env bash
# Run benchmarks, printing results and saving them to results/<benchmark>.txt.
#
# Usage: ./run_benchmarks.sh [synthetic] [pig] [dlpfc] [heart]   (default: all)
#   synthetic: generated tissue, ~5 min
#   pig:       pig esophagus regression test data, seconds
#   dlpfc:     human cortex, ~1 min; needs ./download_data.sh dlpfc first
#   heart:     human heart, ~1 min; needs ./download_data.sh heart first
#
# Uses ../.venv/bin/python if present; set PYTHON to override.
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${PYTHON:-}" ]; then
  if [ -x ../.venv/bin/python ]; then PYTHON=../.venv/bin/python; else PYTHON=python; fi
fi
if [ $# -eq 0 ]; then set -- synthetic pig dlpfc heart; fi

mkdir -p results
for benchmark in "$@"; do
  case $benchmark in
    synthetic | pig) ;;
    dlpfc | heart)
      if [ ! -f "data/$benchmark.csv.gz" ]; then
        echo "data/$benchmark.csv.gz not found: run ./download_data.sh $benchmark first" >&2
        exit 1
      fi
      ;;
    *)
      echo "unknown benchmark: $benchmark (expected synthetic, pig, dlpfc or heart)" >&2
      exit 1
      ;;
  esac
  echo "===== $benchmark"
  "$PYTHON" "$benchmark.py" | tee "results/$benchmark.txt"
done
