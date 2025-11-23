#!/bin/bash
#SBATCH -Jdeepseek-vllm-server
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1 
#SBATCH --mem-per-gpu=224GB
#SBATCH -t4:00:00        # 4-hour runtime. Adjust as needed.
#SBATCH -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences 
#SBATCH --mail-user=mzheng92@gatech.edu # E-mail address for notifications

# --- Setup ---
cd $SLURM_SUBMIT_DIR

# Load essential modules
module load anaconda3
module load cuda

# Activate a conda environment
source activate vllm


# IMPORTANT: Update this path to your project's storage directory
export PROJECT_STORAGE_PATH="/storage/ice1/shared/ece8803cai/team11" #<-- CHECK THIS PATH
export CACHE_PATH="${PROJECT_STORAGE_PATH}/.cache"
mkdir -p $CACHE_PATH

export XDG_CACHE_HOME="${CACHE_PATH}"

# --- Find the Node's IP and a Free Port ---
# Get the name of the node we are running on
NODE_HOSTNAME=$(hostname -f)
# Find a random free port
FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Print the connection details to the output file
echo "----------------------------------------------------------------"
echo "vLLM Server Started on: ${NODE_HOSTNAME}"
echo "vLLM Server Port: ${FREE_PORT}"
echo "Connect to this server using URL: http://${NODE_HOSTNAME}:${FREE_PORT}"
echo "----------------------------------------------------------------"

# --- Launch vLLM Server ---
# This command starts the OpenAI-compatible server.
# --model: The model from HuggingFace
# --host: ${NODE_HOSTNAME} so it's accessible from outside the container
# --port: The free port we found
# --tensor-parallel-size 1: Use 1 GPU
# --dtype auto: Will use bf16 on an A100 for better performance
python -m vllm.entrypoints.openai.api_server \
    --model "deepseek-ai/deepseek-coder-33b-instruct" \
    --host "${NODE_HOSTNAME}" \
    --port "${FREE_PORT}" \
    --tensor-parallel-size 1 \
    --dtype auto \
    --trust-remote-code \
    --max-model-len 16384