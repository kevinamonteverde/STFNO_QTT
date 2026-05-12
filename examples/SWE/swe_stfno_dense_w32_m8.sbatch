#!/bin/bash
#SBATCH --account=m4484
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/stfno_dense_w32_m8/slurm_%j.out
#SBATCH --error=/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/stfno_dense_w32_m8/slurm_%j.err
#SBATCH --job-name=swe_stfno_w32_m8
#SBATCH --mail-user=kevinacosta@lbl.gov
#SBATCH --mail-type=BEGIN,END,FAIL

module load pytorch

mkdir -p /pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/stfno_dense_w32_m8

cd /global/u2/p/pepi/STFNO_QTT/examples/SWE
export PYTHONPATH=/global/u2/p/pepi/STFNO_QTT:$PYTHONPATH
export PYTHONUNBUFFERED=1

python3 main.py \
    --data_path /global/cfs/cdirs/m5031/data/sw2.pkl \
    --output_dir /pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/stfno_dense_w32_m8 \
    --operator_type spectral --width 32 --modes 8 \
    --n_layers 4 \
    --T_in 5 \
    --epochs 200 \
    --batch_size 4 \
    --seed 0
