#!/bin/bash
#SBATCH -J test_vlm
#SBATCH -N 1
#SBATCH --gres=gpu:H100:1
#SBATCH -t 180
#SBATCH --mem=64G
#SBATCH -q coc-ice
#SBATCH -o logs/test_vlm_%j.out

module load anaconda3
conda activate vlm-sched

export HF_HOME=/storage/ice1/shared/ece8803cai/czeng92/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /storage/ice1/shared/ece8803cai/czeng92/vlm_scheduler

nvidia-smi --query-gpu=timestamp,power.draw --format=csv -l 1 > logs/power_test_$SLURM_JOB_ID.log &
POWER_PID=$!

python src/min_test_vlm.py

kill $POWER_PID || true
