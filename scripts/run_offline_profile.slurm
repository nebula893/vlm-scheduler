#!/bin/bash
#SBATCH -J offline_profile
#SBATCH -N 1
#SBATCH --gres=gpu:H100:1
#SBATCH -t 180
#SBATCH --mem=64G
#SBATCH -q coc-ice
#SBATCH -o logs/offline_profile_%j.out

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

mkdir -p logs

/storage/ice1/shared/ece8803cai/czeng92/conda-envs/vlm-sched/bin/python src/offline_profile.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --out data/offline_profile.json
