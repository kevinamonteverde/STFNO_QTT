#!/bin/bash
#SBATCH --account=m4484
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=40:00:00
#SBATCH --output=/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/realop_qtt_r32_w64_400ep/slurm_%j.out
#SBATCH --error=/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/realop_qtt_r32_w64_400ep/slurm_%j.err
#SBATCH --job-name=swe_r32_w64_4
#SBATCH --mail-user=kevinacosta@lbl.gov
#SBATCH --mail-type=BEGIN,END,FAIL

module load pytorch

mkdir -p /pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/realop_qtt_r32_w64_400ep

cd /global/u2/p/pepi/STFNO_QTT/examples/SWE
export PYTHONPATH=/global/u2/p/pepi/STFNO_QTT:$PYTHONPATH
export PYTHONUNBUFFERED=1

python3 main.py \
    --data_path /global/cfs/cdirs/m5031/data/sw2.pkl \
    --output_dir /pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/realop_qtt_r32_w64_400ep \
    --operator_type real_op \
    --factorization_type qtt \
    --factorization_rank 32 \
    --quantize_last_ndims 6 \
    --bit_ordering serial \
    --ft_implementation reconstructed \
    --width 64 \
    --modes 16 \
    --n_layers 4 \
    --T_in 5 \
    --epochs 400 \
    --batch_size 4 \
    --seed 0
