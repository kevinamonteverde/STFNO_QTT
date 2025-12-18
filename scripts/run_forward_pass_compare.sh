#!/usr/bin/env bash
set -e

# Minimal driver: run the NIMROD STFNO example in the current checkout and then
# in the legacy clone with identical arguments.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OLD_DIR="${ROOT_DIR}/STFNO_QTT_old/STFNO_QTT"

if [[ ! -d "${OLD_DIR}" ]]; then
    echo "Legacy checkout not found at ${OLD_DIR}" >&2
    exit 1
fi

module load python

echo "=== current repo ==="
cd "${ROOT_DIR}"
start1=$(date +%s.%N)
python examples/NIMROD3D/main.py \
    --factorization_type tt \
    --factorization_rank 4 \
    --epochs 1 \
    --modes 6 \ 
    --width 8 \
    --output_dir results_tt_compare
end1=$(date +%s.%N)
python - <<PY
start1 = ${start1}
end1 = ${end1}
print(f"current runtime: {end1 - start1:.3f} s")
PY

echo "=== legacy repo ==="
cd "${OLD_DIR}"
start2=$(date +%s.%N)
python examples/NIMROD3D/main.py \
    --factorization_type tt \
    --factorization_rank 4 \
    --epochs 1 \
    --modes 6 \
    --width 8 \
    --output_dir results_tt_compare
end2=$(date +%s.%N)
python - <<PY
start2 = ${start2}
end2 = ${end2}
start1 = ${start1}
end1 = ${end1}
print(f"current runtime: {end1 - start1:.3f} s")
print(f"legacy runtime: {end2 - start2:.3f} s")
PY
