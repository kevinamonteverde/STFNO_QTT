#!/bin/bash
#SBATCH --account=m4484
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/baseline_spectral_dense_w32_m16/rollout_eval_%j.out
#SBATCH --error=/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/baseline_spectral_dense_w32_m16/rollout_eval_%j.err
#SBATCH --job-name=swe_eval_rollout
#SBATCH --mail-user=kevinacosta@lbl.gov
#SBATCH --mail-type=BEGIN,END,FAIL

module load pytorch

cd /global/u2/p/pepi/STFNO_QTT/examples/SWE
export PYTHONPATH=/global/u2/p/pepi/STFNO_QTT:$PYTHONPATH
export PYTHONUNBUFFERED=1

CKPT=/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/baseline_spectral_dense_w32_m16/best_model.pt

echo "=== 20-step rollout ==="
python3 eval_rollout.py \
    --checkpoint $CKPT \
    --operator_type spectral \
    --width 32 --modes 16 --n_layers 4 --T_in 5 \
    --rollout_steps 20

echo ""
echo "=== Full rollout (67 steps) ==="
python3 eval_rollout.py \
    --checkpoint $CKPT \
    --operator_type spectral \
    --width 32 --modes 16 --n_layers 4 --T_in 5 \
    --rollout_steps 67
