#!/bin/bash

#SBATCH --job-name=moe-lm-eval
#SBATCH --account=cse585f25_class
#SBATCH --partition=gpu_mig40
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120g
#SBATCH --gpus=1

module load cuda
module load cudnn
module load uv

echo "[`date '+%Y-%m-%d %H:%M:%S'`] Loaded modules: cuda, cudnn, uv"

uv run python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"

if [ $? -eq 0 ]; then
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] CUDA available for PyTorch"
else
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] Failed to use CUDA for PyTorch, exiting..."
    exit
fi

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME="/scratch/cse585f25_class_root/cse585f25_class/tymiao/.cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

MODEL_NAME="mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_REVISION="main"
DATASET="gsm8k"
NUM_FEWSHOT=0
LIMIT=1
OUTPUT_DIR="results_gsm8k"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${HF_DATASETS_CACHE}"

echo "[`date '+%Y-%m-%d %H:%M:%S'`] Launching moe_eval with model=${MODEL_NAME}, dataset=${DATASET}, limit=${LIMIT}"

uv run python moe_eval.py \
    --model "${MODEL_NAME}" \
    --dataset "${DATASET}" \
    --revision "${MODEL_REVISION}" \
    --limit ${LIMIT} \
    --num_fewshot ${NUM_FEWSHOT} \
    --load_in_4bit \
    --dtype bfloat16 \
    --device_map auto \
    --trust_remote_code \
    --output "${OUTPUT_DIR}/${SLURM_JOB_ID:-manual_run}_metrics.json"

status=$?

if [ ${status} -eq 0 ]; then
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] moe_eval completed successfully."
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] Results saved to ${OUTPUT_DIR}/${SLURM_JOB_ID:-manual_run}_metrics.json"
else
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] moe_eval exited with status ${status}."
fi
