#!/bin/bash
#SBATCH --account=m4484
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/smoke_spectral_dense/slurm_%j.out
#SBATCH --error=/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/smoke_spectral_dense/slurm_%j.err
#SBATCH --job-name=swe_smoke

module load pytorch

cd /global/u2/p/pepi/STFNO_QTT/examples/SWE
export PYTHONPATH=/global/u2/p/pepi/STFNO_QTT:$PYTHONPATH

python3 main.py \
    --data_path /global/cfs/cdirs/m5031/data/sw2.pkl \
    --output_dir /pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/smoke_spectral_dense \
    --operator_type spectral \
    --factorization_type dense \
    --modes 16 \
    --width 32 \
    --n_layers 4 \
    --T_in 5 \
    --epochs 5 \
    --batch_size 4 \
    --seed 0
