#!/bin/bash
#
# JOB SPECIFICATIONS:
#SBATCH --job-name=run_filaments
#SBATCH --partition=bhuv_gpu
#SBATCH --qos=bhuv
#SBATCH --time=28:59:59
#SBATCH --output=out_filament_stack.out
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rmaiya@sas.upenn.edu

echo "NODES=1"

s=$(date)
echo "starting"
echo $s
echo "running the thing"
source ~/.bash_profile
conda activate diffusion
cd /home2/rmaiya/DPS_mass_mapping

export CUDA_VISIBLE_DEVICES=2

TOMO_BIN=1 python get_filament_stack.py
TOMO_BIN=2 python get_filament_stack.py
TOMO_BIN=3 python get_filament_stack.py

#python get_filament_stack.py
#python get_filament_stack.py width
#OMP_NUM_THREADS=1 python get_filament_stack.py shuffle
#python get_filament_stack.py halves
#python get_filament_stack.py excl

e=$(date)
echo "ending"
echo $e        
