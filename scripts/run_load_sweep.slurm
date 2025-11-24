#!/bin/bash
#SBATCH -J vlm_sweep
#SBATCH -N 1
#SBATCH --gres=gpu:H100:1
#SBATCH -t 180
#SBATCH --mem=64G
#SBATCH -q coc-ice
#SBATCH -o logs/slurm/load_sweep_%j.out

module load anaconda3

export PYTHONNOUSERSITE=1

export HF_HOME=/storage/ice1/shared/ece8803cai/czeng92/hf-cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME

export VLLM_CACHE=/storage/ice1/shared/ece8803cai/czeng92/vllm-cache

export FLASHINFER_CACHE=/storage/ice1/shared/ece8803cai/czeng92/flashinfer-cache

mkdir -p "$HF_HOME" "$VLLM_CACHE" "$FLASHINFER_CACHE"

rm -rf ~/.cache/vllm
ln -s "$VLLM_CACHE" ~/.cache/vllm

rm -rf ~/.cache/flashinfer
ln -s "$FLASHINFER_CACHE" ~/.cache/flashinfer

rm -rf ~/.cache/huggingface
ln -s "$HF_HOME" ~/.cache/huggingface

cd /storage/ice1/shared/ece8803cai/czeng92/vlm_scheduler

mkdir -p logs/trace

TRACE=data/azure_trace_sample.csv

echo "Start time: $(date)"
python src/run_trace_load_sweep.py \
    --trace ${TRACE} \
    --max_requests 5000 \
    --out logs/trace/load_sweep.csv
echo "End time: $(date)"
