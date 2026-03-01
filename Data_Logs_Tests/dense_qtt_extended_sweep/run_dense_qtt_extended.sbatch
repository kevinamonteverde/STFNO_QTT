#!/bin/bash
#SBATCH -J dqtt_ext
#SBATCH -A m4484
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 03:30:00
#SBATCH -N 1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-511
#SBATCH -o /pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/dense_qtt_extended_sweep/slurm_%A_%a.out
#SBATCH -e /pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/dense_qtt_extended_sweep/slurm_%A_%a.err
#SBATCH --mail-user=kevinacosta@lbl.gov
#SBATCH --mail-type=END,FAIL

# ============================================================================
# Dense-QTT Extended Sweep
#
# Full grid: rank ∈ {2,4,6,...,32} (16 values) × width ∈ {2,4,6,...,64} (32 values)
# = 512 tasks total.  Tasks that find an existing metrics.csv are skipped (fast).
#
# All output → real_operator_sweep/dense_qtt_r{R}_w{W}/
# Slurm logs → pscratch (no home disk impact)
# ============================================================================

set -eo pipefail
module load pytorch
cd /global/u2/p/pepi/STFNO_QTT/examples/NIMROD3D

PSCRATCH="/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests"
TID=$SLURM_ARRAY_TASK_ID

# Build rank and width arrays (step 2, rank 2-32, width 2-64)
RANKS=()
for r in $(seq 2 2 32); do RANKS+=($r); done    # 16 values

WIDTHS=()
for w in $(seq 2 2 64); do WIDTHS+=($w); done   # 32 values

R_IDX=$(( TID / 32 ))
W_IDX=$(( TID % 32 ))
R=${RANKS[$R_IDX]}
W=${WIDTHS[$W_IDX]}

OUT="$PSCRATCH/real_operator_sweep/dense_qtt_r${R}_w${W}"
mkdir -p "$OUT"

# Skip if this run already completed successfully
if [ -f "$OUT/metrics.csv" ] && [ -s "$OUT/metrics.csv" ]; then
    NLINES=$(wc -l < "$OUT/metrics.csv")
    if [ "$NLINES" -ge 2 ]; then
        echo "Task $TID: SKIP — dense_qtt_r${R}_w${W} already has metrics.csv"
        exit 0
    fi
fi

echo "Task $TID: dense_qtt_r=${R} w=${W} → $OUT"
START=$(date +%s)

srun -n 1 --gpus-per-task=1 -c ${SLURM_CPUS_PER_TASK} \
    python3 main.py \
        --operator_type dense \
        --factorization_type qtt \
        --quantize_last_ndims 8 \
        --factorization_rank $R \
        --width $W \
        --epochs 400 \
        --ft_implementation factorized \
        --output_dir "$OUT" \
        --seed 0 \
    2>&1 | tee "$OUT/training.log"

RUNTIME=$(( $(date +%s) - START ))

FINAL=$(grep "ep= 399" "$OUT/training.log" 2>/dev/null | tail -1)
TRAIN_L2=$(echo "$FINAL" | grep -oP 'train_l2_full / ntrain=\s*\K[\d.e+\-]+')
TEST_L2=$(echo  "$FINAL" | grep -oP 'test_l2_full / ntest=\s*\K[\d.e+\-]+')
PARAMS=$(echo   "$FINAL" | grep -oP 'count_params\(model\)=\s*\K\d+')
TRAIN_T=$(echo  "$FINAL" | grep -oP 't12mid - t1 \(trainTime\)=\s*\K[\d.e+\-]+')
TEST_T=$(echo   "$FINAL" | grep -oP 't2 - t12mid \(testTime\)=\s*\K[\d.e+\-]+')

echo "operator_type,factorization_type,quantize_last_ndims,rank,modes,width,seed,param_count,train_l2,test_l2,train_time_s,test_time_s,total_runtime_s,exit_code" > "$OUT/metrics.csv"
echo "dense,qtt,8,$R,,$W,0,$PARAMS,$TRAIN_L2,$TEST_L2,$TRAIN_T,$TEST_T,$RUNTIME,0" >> "$OUT/metrics.csv"

echo "Task $TID: done. test_l2=$TEST_L2 params=$PARAMS runtime=${RUNTIME}s"
