#!/usr/bin/env bash
# Download the benchmark datasets into data/raw and process them into
# data/<dataset>.csv.gz with prepare_data.py. Files already downloaded are
# skipped, so this can be rerun after an interruption.
#
# Usage: ./download_data.sh [dlpfc] [heart]   (default: both)
#   dlpfc: human cortex, Maynard et al. (2021), ~4 MB
#   heart: human myocardial infarction, Kuppe et al. (2022), ~2.1 GB
#
# Requires curl, and Python with friendlyspots, pandas and h5py (for heart).
# Uses ../.venv/bin/python if present; set PYTHON to override.
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${PYTHON:-}" ]; then
  if [ -x ../.venv/bin/python ]; then PYTHON=../.venv/bin/python; else PYTHON=python; fi
fi
if [ $# -eq 0 ]; then set -- dlpfc heart; fi

fetch() {
  local url=$1 out=$2
  if [ -s "$out" ]; then return; fi
  mkdir -p "$(dirname "$out")"
  echo "downloading $(basename "$out")"
  curl -fL --retry 3 -o "$out.part" "$url"
  mv "$out.part" "$out"
}

for dataset in "$@"; do
  case $dataset in
    dlpfc)
      base=https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/10X
      fetch "$base/barcode_level_layer_map.tsv" data/raw/dlpfc/layer_map.tsv
      for id in 151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676; do
        fetch "$base/$id/tissue_positions_list.txt" "data/raw/dlpfc/${id}_positions.txt"
      done
      ;;
    heart)
      for name in control_P1 control_P7 control_P8 control_P17 RZ_P3 RZ_P6 RZ_P9 RZ_P11 \
                  IZ_P3 IZ_P10 IZ_P15 IZ_P16 FZ_P14 FZ_P18 FZ_P20; do
        fetch "https://zenodo.org/api/records/6578047/files/Visium_$name.h5ad/content" \
              "data/raw/heart/Visium_$name.h5ad"
      done
      ;;
    *)
      echo "unknown dataset: $dataset (expected dlpfc or heart)" >&2
      exit 1
      ;;
  esac
done

"$PYTHON" prepare_data.py "$@"
