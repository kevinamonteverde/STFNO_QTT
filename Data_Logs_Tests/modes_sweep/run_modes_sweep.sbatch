#!/bin/bash
#SBATCH -J modes_sweep
#SBATCH -A m4484
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 03:30:00
#SBATCH -N 1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-15
#SBATCH -o /pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/modes_sweep/slurm_%A_%a.out
#SBATCH -e /pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/modes_sweep/slurm_%A_%a.err
#SBATCH --mail-user=kevinacosta@lbl.gov
#SBATCH --mail-type=END,FAIL

# ============================================================================
# Modes Sweep — fig04 modes sensitivity for QTT3, QTT5, and Unfactorized
#
# tasks  0-5  : Spectral-QTT3  r=12 w=32 at m=4,6,8,10,14,16
#               → spectral_qtt_q3_sweep/spectral_qtt_q3_r12_m{M}_w32/
# tasks  6-11 : Spectral-QTT5  r=12 w=32 at m=4,6,8,10,14,16
#               → spectral_qtt_q5_sweep/spectral_qtt_q5_r12_m{M}_w32/
# tasks 12-15 : Unfactorized   at m=14,16 × w=8,32
#               → unfactorized_width_sweep/dense_m{M}_w{W}/
#
# m=12 already exists for QTT3/QTT5 from the main sweeps — not rerun here.
# Unfactorized m=4..12 already exists in stfno_sweep/combined_results.csv.
# ============================================================================

set -eo pipefail
module load pytorch
cd /global/u2/p/pepi/STFNO_QTT/examples/NIMROD3D

PSCRATCH="/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests"
MODES=(4 6 8 10 14 16)

if [ "$SLURM_ARRAY_TASK_ID" -lt 6 ]; then
    # ── Spectral-QTT3 r=12 w=32, varying modes ──────────────────────────────
    M=${MODES[$SLURM_ARRAY_TASK_ID]}
    OUT="$PSCRATCH/spectral_qtt_q3_sweep/spectral_qtt_q3_r12_m${M}_w32"
    mkdir -p "$OUT"
    echo "Running: Spectral-QTT3 r=12 m=$M w=32"
    echo "Output Directory: $OUT"
    START=$(date +%s)
    srun -n 1 --gpus-per-task=1 -c ${SLURM_CPUS_PER_TASK} \
        python3 main.py --operator_type spectral --factorization_type qtt \
            --quantize_last_ndims 3 --factorization_rank 12 --modes $M --width 32 --epochs 400 \
            --ft_implementation factorized --output_dir "$OUT" --seed 0 \
        2>&1 | tee "$OUT/training.log"
    END=$(date +%s)
    RUNTIME=$((END - START))
    FINAL=$(grep "ep= 399" "$OUT/training.log" | tail -1)
    TRAIN_L2=$(echo "$FINAL" | grep -oP 'train_l2_full / ntrain=\s*\K[\d.e+\-]+')
    TEST_L2=$(echo  "$FINAL" | grep -oP 'test_l2_full / ntest=\s*\K[\d.e+\-]+')
    PARAMS=$(echo   "$FINAL" | grep -oP 'count_params\(model\)=\s*\K\d+')
    TRAIN_T=$(echo  "$FINAL" | grep -oP 't12mid - t1 \(trainTime\)=\s*\K[\d.e+\-]+')
    TEST_T=$(echo   "$FINAL" | grep -oP 't2 - t12mid \(testTime\)=\s*\K[\d.e+\-]+')
    echo "operator_type,factorization_type,quantize_last_ndims,rank,modes,width,seed,param_count,train_l2,test_l2,train_time_s,test_time_s,total_runtime_s,exit_code" > "$OUT/metrics.csv"
    echo "spectral,qtt,3,12,$M,32,0,$PARAMS,$TRAIN_L2,$TEST_L2,$TRAIN_T,$TEST_T,$RUNTIME,0" >> "$OUT/metrics.csv"

elif [ "$SLURM_ARRAY_TASK_ID" -lt 12 ]; then
    # ── Spectral-QTT5 r=12 w=32, varying modes ──────────────────────────────
    IDX=$((SLURM_ARRAY_TASK_ID - 6))
    M=${MODES[$IDX]}
    OUT="$PSCRATCH/spectral_qtt_q5_sweep/spectral_qtt_q5_r12_m${M}_w32"
    mkdir -p "$OUT"
    echo "Running: Spectral-QTT5 r=12 m=$M w=32"
    echo "Output Directory: $OUT"
    START=$(date +%s)
    srun -n 1 --gpus-per-task=1 -c ${SLURM_CPUS_PER_TASK} \
        python3 main.py --operator_type spectral --factorization_type qtt \
            --quantize_last_ndims 5 --factorization_rank 12 --modes $M --width 32 --epochs 400 \
            --ft_implementation factorized --output_dir "$OUT" --seed 0 \
        2>&1 | tee "$OUT/training.log"
    END=$(date +%s)
    RUNTIME=$((END - START))
    FINAL=$(grep "ep= 399" "$OUT/training.log" | tail -1)
    TRAIN_L2=$(echo "$FINAL" | grep -oP 'train_l2_full / ntrain=\s*\K[\d.e+\-]+')
    TEST_L2=$(echo  "$FINAL" | grep -oP 'test_l2_full / ntest=\s*\K[\d.e+\-]+')
    PARAMS=$(echo   "$FINAL" | grep -oP 'count_params\(model\)=\s*\K\d+')
    TRAIN_T=$(echo  "$FINAL" | grep -oP 't12mid - t1 \(trainTime\)=\s*\K[\d.e+\-]+')
    TEST_T=$(echo   "$FINAL" | grep -oP 't2 - t12mid \(testTime\)=\s*\K[\d.e+\-]+')
    echo "operator_type,factorization_type,quantize_last_ndims,rank,modes,width,seed,param_count,train_l2,test_l2,train_time_s,test_time_s,total_runtime_s,exit_code" > "$OUT/metrics.csv"
    echo "spectral,qtt,5,12,$M,32,0,$PARAMS,$TRAIN_L2,$TEST_L2,$TRAIN_T,$TEST_T,$RUNTIME,0" >> "$OUT/metrics.csv"

else
    # ── Unfactorized m=14,16 at w=8,32 — extend existing stfno_sweep data ──
    IDX=$((SLURM_ARRAY_TASK_ID - 12))
    UF_MODES=(14 14 16 16)
    UF_WIDTHS=(8 32 8 32)
    M=${UF_MODES[$IDX]}
    W=${UF_WIDTHS[$IDX]}
    OUT="$PSCRATCH/unfactorized_width_sweep/dense_m${M}_w${W}"
    mkdir -p "$OUT"
    echo "Running: Unfactorized m=$M w=$W"
    echo "Output Directory: $OUT"
    START=$(date +%s)
    srun -n 1 --gpus-per-task=1 -c ${SLURM_CPUS_PER_TASK} \
        python3 main.py --operator_type spectral --factorization_type dense \
            --modes $M --width $W --epochs 400 \
            --output_dir "$OUT" --seed 0 \
        2>&1 | tee "$OUT/training.log"
    END=$(date +%s)
    RUNTIME=$((END - START))
    FINAL=$(grep "ep= 399" "$OUT/training.log" | tail -1)
    TRAIN_L2=$(echo "$FINAL" | grep -oP 'train_l2_full / ntrain=\s*\K[\d.e+\-]+')
    TEST_L2=$(echo  "$FINAL" | grep -oP 'test_l2_full / ntest=\s*\K[\d.e+\-]+')
    PARAMS=$(echo   "$FINAL" | grep -oP 'count_params\(model\)=\s*\K\d+')
    TRAIN_T=$(echo  "$FINAL" | grep -oP 't12mid - t1 \(trainTime\)=\s*\K[\d.e+\-]+')
    TEST_T=$(echo   "$FINAL" | grep -oP 't2 - t12mid \(testTime\)=\s*\K[\d.e+\-]+')
    echo "operator_type,factorization_type,quantize_last_ndims,rank,modes,width,seed,param_count,train_l2,test_l2,train_time_s,test_time_s,total_runtime_s,exit_code" > "$OUT/metrics.csv"
    echo "spectral,dense,,,$M,$W,0,$PARAMS,$TRAIN_L2,$TEST_L2,$TRAIN_T,$TEST_T,$RUNTIME,0" >> "$OUT/metrics.csv"
fi
