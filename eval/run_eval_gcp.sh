#!/bin/bash

#SBATCH --job-name=lm-eval-mixtral
#SBATCH --account=cse585f25_class
#SBATCH --partition=gpu_mig40
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120g
#SBATCH --gpus=1

# module load cuda
# module load cudnn
# module load uv

# echo "[`date '+%Y-%m-%d %H:%M:%S'`] Loaded modules: cuda, cudnn, uv"

# uv run python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"

# if [ $? -eq 0 ]; then
#     echo "[`date '+%Y-%m-%d %H:%M:%S'`] CUDA available for PyTorch"
# else
#     echo "[`date '+%Y-%m-%d %H:%M:%S'`] Failed to use CUDA for PyTorch, exiting..."
#     exit
# fi

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
# export HF_HOME="/scratch/cse585f25_class_root/cse585f25_class/tymiao/.cache"
# export HF_DATASETS_CACHE="${HF_HOME}/datasets"

MODEL_NAME="deepseek-ai/DeepSeek-V2-Lite"
TASKS="gsm8k"
BATCH_SIZE=1
LIMIT=256
NUM_FEWSHOT=5
# mkdir -p "${HF_HOME}"
# mkdir -p "${HF_DATASETS_CACHE}"

echo "[`date '+%Y-%m-%d %H:%M:%S'`] Running lm_eval with model=${MODEL_NAME}, tasks=${TASKS}"

python3 run_lm_eval.py \
    --model hf \
    --model_args "pretrained=${MODEL_NAME}" \
    --tasks "${TASKS}" \
    --batch_size ${BATCH_SIZE} \
    --limit ${LIMIT} \
    --num_fewshot ${NUM_FEWSHOT}

status=$?

if [ ${status} -eq 0 ]; then
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] lm_eval completed successfully."
else
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] lm_eval exited with status ${status}."
fi
