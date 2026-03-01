#!/bin/bash
#SBATCH -J qtt3_m16_rerun
#SBATCH -A m4484
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 03:30:00
#SBATCH -N 1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --array=1-3
#SBATCH -o /pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/qtt3_m16_rerun/slurm_%A_%a.out
#SBATCH -e /pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/qtt3_m16_rerun/slurm_%A_%a.err
#SBATCH --mail-user=kevinacosta@lbl.gov
#SBATCH --mail-type=END,FAIL

# ============================================================================
# QTT3 m=16 rerun — seed investigation
#
# Original seed=0 run plateaued early at test_l2~0.0173 vs m=14's 0.0110.
# Re-running with seeds 1,2,3 to determine if this is a seed artifact
# or a genuine structural issue with QTT3 at modes=16 (no QTT padding).
#
# Output → pscratch/qtt3_m16_rerun/seed{S}/
# ============================================================================

set -eo pipefail
module load pytorch
cd /global/u2/p/pepi/STFNO_QTT/examples/NIMROD3D

PSCRATCH="/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests"
S=$SLURM_ARRAY_TASK_ID   # 1, 2, 3

OUT="$PSCRATCH/qtt3_m16_rerun/seed${S}"
mkdir -p "$OUT"

echo "Task $S: QTT3 r=12 m=16 w=32 seed=$S → $OUT"
START=$(date +%s)

srun -n 1 --gpus-per-task=1 -c ${SLURM_CPUS_PER_TASK} \
    python3 main.py \
        --operator_type spectral \
        --factorization_type qtt \
        --quantize_last_ndims 3 \
        --factorization_rank 12 \
        --modes 16 \
        --width 32 \
        --epochs 400 \
        --ft_implementation factorized \
        --output_dir "$OUT" \
        --seed $S \
    2>&1 | tee "$OUT/training.log"

RUNTIME=$(( $(date +%s) - START ))

FINAL=$(grep "ep= 399" "$OUT/training.log" 2>/dev/null | tail -1)
TEST_L2=$(echo "$FINAL" | grep -oP 'test_l2_full / ntest=\s*\K[\d.e+\-]+')
TRAIN_L2=$(echo "$FINAL" | grep -oP 'train_l2_full / ntrain=\s*\K[\d.e+\-]+')

echo "operator_type,factorization_type,quantize_last_ndims,rank,modes,width,seed,param_count,train_l2,test_l2,train_time_s,test_time_s,total_runtime_s,exit_code" > "$OUT/metrics.csv"
echo "spectral,qtt,3,12,16,32,$S,807946,$TRAIN_L2,$TEST_L2,,,${RUNTIME},0" >> "$OUT/metrics.csv"

echo "Task $S done: seed=$S test_l2=$TEST_L2 runtime=${RUNTIME}s"
echo "Compare: seed=0 got test_l2=0.01730 (plateaued)"
