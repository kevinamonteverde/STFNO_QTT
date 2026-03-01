#!/bin/bash
#SBATCH -J gap_fill
#SBATCH -A m4484
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 03:30:00
#SBATCH -N 1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-97
#SBATCH -o /pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/gap_fill_sweep/slurm_%A_%a.out
#SBATCH -e /pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/gap_fill_sweep/slurm_%A_%a.err
#SBATCH --mail-user=kevinacosta@lbl.gov
#SBATCH --mail-type=END,FAIL

# ============================================================================
# Gap-Fill Sweep — completes unified_results.csv coverage
#
# Tasks  0-17 : Spectral-TT   r=12, m=4,6,8,10,14,16 × w=8,16,24   (18 tasks)
#               → spectral_tt_modes_sweep/spectral_tt_r12_m{M}_w{W}/
#               Purpose: fills shaded band in figH at non-w32 widths
#
# Tasks 18-35 : Spectral-QTT3 r=12, m=4,6,8,10,14,16 × w=8,16,24   (18 tasks)
#               → spectral_qtt_q3_sweep/spectral_qtt_q3_r12_m{M}_w{W}/
#
# Tasks 36-53 : Spectral-QTT5 r=12, m=4,6,8,10,14,16 × w=8,16,24   (18 tasks)
#               → spectral_qtt_q5_sweep/spectral_qtt_q5_r12_m{M}_w{W}/
#
# Tasks 54-97 : Spectral-QTT5 rank fill r=1,3,5,6,7,9,10,11,13,14,15 × w=8,16,24,32  (44 tasks)
#               → spectral_qtt_q5_sweep/spectral_qtt_q5_r{R}_m12_w{W}/
#               Purpose: matches full rank coverage already present in TT and QTT3
# ============================================================================

set -eo pipefail
module load pytorch
cd /global/u2/p/pepi/STFNO_QTT/examples/NIMROD3D

PSCRATCH="/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests"
TID=$SLURM_ARRAY_TASK_ID

MODES_LIST=(4 6 8 10 14 16)
WIDTHS_LIST=(8 16 24)

# Helper: write metrics.csv from last ep=399 line in training.log
write_metrics() {
    local OUT="$1" OP="$2" FT="$3" QND="$4" R="$5" M="$6" W="$7"
    local FINAL TRAIN_L2 TEST_L2 PARAMS TRAIN_T TEST_T RUNTIME="$8"
    FINAL=$(grep "ep= 399" "$OUT/training.log" 2>/dev/null | tail -1)
    TRAIN_L2=$(echo "$FINAL" | grep -oP 'train_l2_full / ntrain=\s*\K[\d.e+\-]+')
    TEST_L2=$(echo  "$FINAL" | grep -oP 'test_l2_full / ntest=\s*\K[\d.e+\-]+')
    PARAMS=$(echo   "$FINAL" | grep -oP 'count_params\(model\)=\s*\K\d+')
    TRAIN_T=$(echo  "$FINAL" | grep -oP 't12mid - t1 \(trainTime\)=\s*\K[\d.e+\-]+')
    TEST_T=$(echo   "$FINAL" | grep -oP 't2 - t12mid \(testTime\)=\s*\K[\d.e+\-]+')
    echo "operator_type,factorization_type,quantize_last_ndims,rank,modes,width,seed,param_count,train_l2,test_l2,train_time_s,test_time_s,total_runtime_s,exit_code" > "$OUT/metrics.csv"
    echo "$OP,$FT,$QND,$R,$M,$W,0,$PARAMS,$TRAIN_L2,$TEST_L2,$TRAIN_T,$TEST_T,$RUNTIME,0" >> "$OUT/metrics.csv"
}

# ── Tasks 0-17 : Spectral-TT modes at w=8,16,24 ──────────────────────────────
if [ "$TID" -lt 18 ]; then
    W_IDX=$(( TID / 6 ))
    M_IDX=$(( TID % 6 ))
    W=${WIDTHS_LIST[$W_IDX]}
    M=${MODES_LIST[$M_IDX]}
    OUT="$PSCRATCH/spectral_tt_modes_sweep/spectral_tt_r12_m${M}_w${W}"
    mkdir -p "$OUT"
    echo "Task $TID: Spectral-TT r=12 m=$M w=$W → $OUT"
    START=$(date +%s)
    srun -n 1 --gpus-per-task=1 -c ${SLURM_CPUS_PER_TASK} \
        python3 main.py --operator_type spectral --factorization_type tt \
            --factorization_rank 12 --modes $M --width $W --epochs 400 \
            --ft_implementation factorized --output_dir "$OUT" --seed 0 \
        2>&1 | tee "$OUT/training.log"
    RUNTIME=$(( $(date +%s) - START ))
    write_metrics "$OUT" spectral tt "" 12 $M $W $RUNTIME

# ── Tasks 18-35 : Spectral-QTT3 modes at w=8,16,24 ──────────────────────────
elif [ "$TID" -lt 36 ]; then
    IDX=$(( TID - 18 ))
    W_IDX=$(( IDX / 6 ))
    M_IDX=$(( IDX % 6 ))
    W=${WIDTHS_LIST[$W_IDX]}
    M=${MODES_LIST[$M_IDX]}
    OUT="$PSCRATCH/spectral_qtt_q3_sweep/spectral_qtt_q3_r12_m${M}_w${W}"
    mkdir -p "$OUT"
    echo "Task $TID: Spectral-QTT3 r=12 m=$M w=$W → $OUT"
    START=$(date +%s)
    srun -n 1 --gpus-per-task=1 -c ${SLURM_CPUS_PER_TASK} \
        python3 main.py --operator_type spectral --factorization_type qtt \
            --quantize_last_ndims 3 --factorization_rank 12 --modes $M --width $W --epochs 400 \
            --ft_implementation factorized --output_dir "$OUT" --seed 0 \
        2>&1 | tee "$OUT/training.log"
    RUNTIME=$(( $(date +%s) - START ))
    write_metrics "$OUT" spectral qtt 3 12 $M $W $RUNTIME

# ── Tasks 36-53 : Spectral-QTT5 modes at w=8,16,24 ──────────────────────────
elif [ "$TID" -lt 54 ]; then
    IDX=$(( TID - 36 ))
    W_IDX=$(( IDX / 6 ))
    M_IDX=$(( IDX % 6 ))
    W=${WIDTHS_LIST[$W_IDX]}
    M=${MODES_LIST[$M_IDX]}
    OUT="$PSCRATCH/spectral_qtt_q5_sweep/spectral_qtt_q5_r12_m${M}_w${W}"
    mkdir -p "$OUT"
    echo "Task $TID: Spectral-QTT5 r=12 m=$M w=$W → $OUT"
    START=$(date +%s)
    srun -n 1 --gpus-per-task=1 -c ${SLURM_CPUS_PER_TASK} \
        python3 main.py --operator_type spectral --factorization_type qtt \
            --quantize_last_ndims 5 --factorization_rank 12 --modes $M --width $W --epochs 400 \
            --ft_implementation factorized --output_dir "$OUT" --seed 0 \
        2>&1 | tee "$OUT/training.log"
    RUNTIME=$(( $(date +%s) - START ))
    write_metrics "$OUT" spectral qtt 5 12 $M $W $RUNTIME

# ── Tasks 54-97 : Spectral-QTT5 rank fill × w=8,16,24,32 ────────────────────
else
    IDX=$(( TID - 54 ))
    FILL_RANKS=(1 3 5 6 7 9 10 11 13 14 15)
    FILL_WIDTHS=(8 16 24 32)
    R_IDX=$(( IDX / 4 ))
    W_IDX=$(( IDX % 4 ))
    R=${FILL_RANKS[$R_IDX]}
    W=${FILL_WIDTHS[$W_IDX]}
    OUT="$PSCRATCH/spectral_qtt_q5_sweep/spectral_qtt_q5_r${R}_m12_w${W}"
    mkdir -p "$OUT"
    echo "Task $TID: Spectral-QTT5 r=$R m=12 w=$W → $OUT"
    START=$(date +%s)
    srun -n 1 --gpus-per-task=1 -c ${SLURM_CPUS_PER_TASK} \
        python3 main.py --operator_type spectral --factorization_type qtt \
            --quantize_last_ndims 5 --factorization_rank $R --modes 12 --width $W --epochs 400 \
            --ft_implementation factorized --output_dir "$OUT" --seed 0 \
        2>&1 | tee "$OUT/training.log"
    RUNTIME=$(( $(date +%s) - START ))
    write_metrics "$OUT" spectral qtt 5 $R 12 $W $RUNTIME
fi
