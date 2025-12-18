#!/bin/bash
#SBATCH -J tt_vs_dense_recon
#SBATCH -o tt_vs_dense_recon.%j.out
#SBATCH -e tt_vs_dense_recon.%j.err
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --time=02:00:00
#SBATCH --partition=regular
#SBATCH --account=m4484
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=kevinacosta@lbl.gov

set -euo pipefail

module load python

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLE_MAIN="$REPO_ROOT/examples/NIMROD3D/main.py"

EPOCHS=5
SPATIAL=32
FACTORIZATION_RANK=4
BATCH_SIZES=(1 2 4)
MODES_LIST=(6 10)
WIDTH_LIST=(8 16)

run_case() {
  local impl="$1"   # factorized (core-by-core TT) or reconstructed (dense materialization)
  local bs="$2"     # batch size
  local modes="$3"
  local width="$4"
  echo "===== Running impl=${impl}, batch_size=${bs}, modes=${modes}, width=${width} ====="
  srun python "$EXAMPLE_MAIN" \
    --factorization_type tt \
    --ft_implementation "$impl" \
    --factorization_rank "$FACTORIZATION_RANK" \
    --epochs "$EPOCHS" \
    --modes "$modes" \
    --width "$width" \
    --spatial_size "$SPATIAL" \
    --batch_size "$bs" \
    --output_dir "$REPO_ROOT/results_tt_vs_dense_${impl}_bs${bs}_m${modes}_w${width}"
}

for modes in "${MODES_LIST[@]}"; do
  for width in "${WIDTH_LIST[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
      run_case factorized "$bs" "$modes" "$width"    # core-by-core tt_contraction path
      run_case reconstructed "$bs" "$modes" "$width" # dense reconstruction path
    done
  done
done

echo "Done."
