#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=synthetic_data_generation
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err

source /scratch/hakeem.at/Queryable-Shared-Reference-Repository/.venv/bin/activate

python synthetic_data_generation_prompt_thresholding.py