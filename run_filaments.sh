#!/bin/bash
#
# JOB SPECIFICATIONS:
#SBATCH --job-name=run_filaments
#SBATCH --partition=bhuv_gpu
#SBATCH --qos=bhuv
#SBATCH --time=28:59:59
#SBATCH --output=out_filaments.out
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
python filaments.py

e=$(date)
echo "ending"
echo $e        
