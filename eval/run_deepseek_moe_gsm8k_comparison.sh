#!/bin/bash

#SBATCH --job-name=deepseek-moe-gsm8k-comparison
#SBATCH --account=cse585f25_class
#SBATCH --partition=gpu_mig40
#SBATCH --time=04:00:00
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

MODEL_NAME="deepseek-ai/deepseek-moe-16b-base"
TASKS="gsm8k"
BATCH_SIZE=16
LIMIT=500
TRACE_DATASET="gsm8k"
OUTPUT_ROOT="runs/deepseek_moe_gsm8k_comparison"

mkdir -p "${HF_HOME}"
mkdir -p "${HF_DATASETS_CACHE}"
mkdir -p "${OUTPUT_ROOT}"

echo "[`date '+%Y-%m-%d %H:%M:%S'`] Starting DeepSeek-MoE-16B GSM8K comparison evaluation"

# -----------------------------------------------------
# 1. Baseline (unmodified model)
# -----------------------------------------------------
echo "[`date '+%Y-%m-%d %H:%M:%S'`] Running baseline evaluation (unmodified model)"

uv run python pipeline.py \
    --model_name "${MODEL_NAME}" \
    --output_root "${OUTPUT_ROOT}/baseline" \
    --trace_dataset "${TRACE_DATASET}" \
    --trace_nsamples 128 \
    --trace_seqlen 2048 \
    --trace_batch_size 1 \
    --eval_tasks "${TASKS}" \
    --eval_batch_size ${BATCH_SIZE} \
    --eval_limit ${LIMIT} \
    --skip_quant \
    --skip_prune

baseline_status=$?
if [ ${baseline_status} -eq 0 ]; then
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] Baseline evaluation completed successfully."
else
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] Baseline evaluation failed with status ${baseline_status}."
    exit ${baseline_status}
fi

# -----------------------------------------------------
# 2. Pruned (1 expert removed)
# -----------------------------------------------------
echo "[`date '+%Y-%m-%d %H:%M:%S'`] Running pruned evaluation (1 expert removed)"

uv run python pipeline.py \
    --model_name "${MODEL_NAME}" \
    --output_root "${OUTPUT_ROOT}/pruned_1expert" \
    --trace_dataset "${TRACE_DATASET}" \
    --trace_nsamples 128 \
    --trace_seqlen 2048 \
    --trace_batch_size 1 \
    --eval_tasks "${TASKS}" \
    --eval_batch_size ${BATCH_SIZE} \
    --eval_limit ${LIMIT} \
    --prune_strategy bottom_k \
    --prune_k 1 \
    --skip_quant \
    --skip_trace \
    --reuse_trace_data

pruned_status=$?
if [ ${pruned_status} -eq 0 ]; then
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] Pruned evaluation completed successfully."
else
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] Pruned evaluation failed with status ${pruned_status}."
    exit ${pruned_status}
fi

# -----------------------------------------------------
# 3. Summary
# -----------------------------------------------------
echo "[`date '+%Y-%m-%d %H:%M:%S'`] ===== Evaluation Summary ====="
echo "Model: ${MODEL_NAME}"
echo "Task: ${TASKS}"
echo "Samples: ${LIMIT}"
echo "Results saved in: ${OUTPUT_ROOT}"
echo "  - Baseline results: ${OUTPUT_ROOT}/baseline/"
echo "  - Pruned results: ${OUTPUT_ROOT}/pruned_1expert/"
echo "[`date '+%Y-%m-%d %H:%M:%S'`] All evaluations completed successfully!"
