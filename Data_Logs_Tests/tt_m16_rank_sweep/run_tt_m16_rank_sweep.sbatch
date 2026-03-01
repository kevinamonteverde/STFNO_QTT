#!/bin/bash
#SBATCH -J tt_m16_ranks
#SBATCH -A m4484
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 03:30:00
#SBATCH -N 1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-1
#SBATCH -o /pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/tt_m16_rank_sweep/slurm_%A_%a.out
#SBATCH -e /pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/tt_m16_rank_sweep/slurm_%A_%a.err
#SBATCH --mail-user=kevinacosta@lbl.gov
#SBATCH --mail-type=END,FAIL

# ============================================================================
# Spectral-TT m=16 rank sweep: r=14 and r=15
#
# Motivation: rank sweep (m=12) shows r=14/15 beat r=12 (0.009025/0.009000
# vs 0.009332). Modes sweep (r=12) shows m=16 beats m=12 (0.008552 vs
# 0.009332). These axes have never been crossed — r=14/15 at m=16 may be
# the true best Spectral-TT config.
#
# Task 0 → r=14, m=16, w=32
# Task 1 → r=15, m=16, w=32
#
# Output → spectral_tt_modes_sweep/spectral_tt_r{R}_m16_w32/
# ============================================================================

set -eo pipefail
module load pytorch
cd /global/u2/p/pepi/STFNO_QTT/examples/NIMROD3D

PSCRATCH="/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests"
RANKS=(14 15)
R=${RANKS[$SLURM_ARRAY_TASK_ID]}

OUT="$PSCRATCH/spectral_tt_modes_sweep/spectral_tt_r${R}_m16_w32"
mkdir -p "$OUT"

echo "Task $SLURM_ARRAY_TASK_ID: Spectral-TT r=$R m=16 w=32 → $OUT"
START=$(date +%s)

srun -n 1 --gpus-per-task=1 -c ${SLURM_CPUS_PER_TASK} \
    python3 main.py \
        --operator_type spectral \
        --factorization_type tt \
        --factorization_rank $R \
        --modes 16 \
        --width 32 \
        --epochs 400 \
        --ft_implementation factorized \
        --output_dir "$OUT" \
        --seed 0 \
    2>&1 | tee "$OUT/training.log"

RUNTIME=$(( $(date +%s) - START ))

FINAL=$(grep "ep= 399" "$OUT/training.log" 2>/dev/null | tail -1)
TEST_L2=$(echo "$FINAL" | grep -oP 'test_l2_full / ntest=\s*\K[\d.e+\-]+')
TRAIN_L2=$(echo "$FINAL" | grep -oP 'train_l2_full / ntrain=\s*\K[\d.e+\-]+')
PARAMS=$(echo "$FINAL" | grep -oP 'count_params\(model\)=\s*\K\d+')

echo "operator_type,factorization_type,rank,modes,width,param_count,train_l2,test_l2,train_time_s,test_time_s,total_runtime_s,exit_code" > "$OUT/metrics.csv"
echo "spectral,tt,$R,16,32,$PARAMS,$TRAIN_L2,$TEST_L2,,,$RUNTIME,0" >> "$OUT/metrics.csv"

echo "Task $SLURM_ARRAY_TASK_ID done: r=$R m=16 test_l2=$TEST_L2 runtime=${RUNTIME}s"
echo "Compare: r=12 m=16 → 0.008552 | r=15 m=12 → 0.009000"
