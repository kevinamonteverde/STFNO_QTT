#!/bin/bash
#SBATCH --account=m4484
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/rollout_eval_realop_all_%j.out
#SBATCH --error=/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/rollout_eval_realop_all_%j.err
#SBATCH --job-name=swe_rollout_realop
#SBATCH --mail-user=kevinacosta@lbl.gov
#SBATCH --mail-type=BEGIN,END,FAIL

module load pytorch

cd /global/u2/p/pepi/STFNO_QTT/examples/SWE
export PYTHONPATH=/global/u2/p/pepi/STFNO_QTT:$PYTHONPATH
export PYTHONUNBUFFERED=1

BASE=/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results

run_rollout() {
    local label=$1
    local ckpt=$2
    local rank=$3
    local width=$4

    echo "=============================="
    echo "CONFIG: $label  (r=$rank w=$width)"
    echo "=============================="

    echo "--- 20-step rollout ---"
    python3 eval_rollout.py \
        --checkpoint "$ckpt" \
        --operator_type real_op \
        --factorization_type qtt \
        --factorization_rank "$rank" \
        --quantize_last_ndims 6 \
        --bit_ordering serial \
        --ft_implementation reconstructed \
        --width "$width" --modes 16 --n_layers 4 --T_in 5 \
        --rollout_steps 20

    echo ""
    echo "--- 67-step rollout ---"
    python3 eval_rollout.py \
        --checkpoint "$ckpt" \
        --operator_type real_op \
        --factorization_type qtt \
        --factorization_rank "$rank" \
        --quantize_last_ndims 6 \
        --bit_ordering serial \
        --ft_implementation reconstructed \
        --width "$width" --modes 16 --n_layers 4 --T_in 5 \
        --rollout_steps 67

    echo ""
}

run_rollout "r32_w64_400ep" "$BASE/realop_qtt_r32_w64_400ep/best_model.pt" 32 64
run_rollout "r24_w64_400ep" "$BASE/realop_qtt_r24_w64_400ep/best_model.pt" 24 64
run_rollout "r16_w64_200ep" "$BASE/realop_qtt_r16_w64/best_model.pt"        16 64
run_rollout "r32_w32_400ep" "$BASE/realop_qtt_r32_w32_400ep/best_model.pt"  32 32
run_rollout "r24_w32_400ep" "$BASE/realop_qtt_r24_w32_400ep/best_model.pt"  24 32
run_rollout "r16_w32_400ep" "$BASE/realop_qtt_r16_w32_400ep/best_model.pt"  16 32
run_rollout "r8_w32_400ep"  "$BASE/realop_qtt_r8_w32_400ep/best_model.pt"    8 32

echo "All rollout evals complete."
