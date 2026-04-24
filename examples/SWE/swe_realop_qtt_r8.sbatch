#!/bin/bash
#SBATCH --account=m4484
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/realop_qtt_r8_w32/slurm_%j.out
#SBATCH --error=/pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/realop_qtt_r8_w32/slurm_%j.err
#SBATCH --job-name=swe_realop_r8
#SBATCH --mail-user=kevinacosta@lbl.gov
#SBATCH --mail-type=BEGIN,END

module load pytorch

cd /global/u2/p/pepi/STFNO_QTT/examples/SWE
export PYTHONPATH=/global/u2/p/pepi/STFNO_QTT:$PYTHONPATH
export PYTHONUNBUFFERED=1

# ST-NO-2D: direct spatial operator, QTT-factorized 6D weight
# Weight shape: (Cin=32, H_in=256, W_in=256, Cout=32, H_out=256, W_out=256)
# TT chain: 5+8+8+5+8+8 = 42 cores (same length as NIMROD 3D at w=32)
python3 main.py \
    --data_path /global/cfs/cdirs/m5031/data/sw2.pkl \
    --output_dir /pscratch/sd/p/pepi/STFNO_QTT/Data_Logs_Tests/swe_results/realop_qtt_r8_w32 \
    --operator_type real_op \
    --factorization_type qtt \
    --factorization_rank 8 \
    --quantize_last_ndims 6 \
    --bit_ordering serial \
    --ft_implementation reconstructed \
    --width 32 \
    --n_layers 4 \
    --T_in 5 \
    --epochs 200 \
    --batch_size 4 \
    --seed 0
