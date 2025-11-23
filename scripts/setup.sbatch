#!/bin/bash
#SBATCH -J Azure_trace         # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 960                    # Duration of the job (Ex: 720 mins = 12 hours)
#SBATCH --mem-per-cpu=10G         # Increased memory for RL training
#SBATCH -q coc-ice
#SBATCH -o logs/ppo/Report-%j.out        # Output to logs directory (relative to project root)
#SBATCH --mail-type=BEGIN,END,FAIL   # Mail preferences
#SBATCH --mail-user=mzheng92@gatech.edu  # E-mail address for notifications

cd /home/hice1/yli3776/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker

# Load modules
module load anaconda3

# Activate conda environment
conda activate cs6220

# Set HuggingFace cache to storage directory (avoid memory issue)
export HF_HOME=/home/hice1/yli3776/.cache/huggingface
export TRANSFORMERS_CACHE=/home/hice1/yli3776/.cache/huggingface/transformers
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE

# Create necessary directories
mkdir -p ../logs/ppo
mkdir -p ../ppo/ppo_checkpoints
mkdir -p ../ppo/rl_adapters

# Show environment info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
nvidia-smi

# Run PPO training from project root so `ppo` is an importable package
echo "Starting PPO training (module mode)..."
python -m ppo.train_ppo