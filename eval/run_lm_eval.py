#!/usr/bin/env python3
"""
Basic script to run lm_eval.
"""

import argparse
import json
from datetime import datetime
import lm_eval
import re
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="vllm")
parser.add_argument("--model_args", type=str, required=True)
parser.add_argument("--tasks", type=str, required=True)
parser.add_argument("--batch_size", type=str, default="auto")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument(
    "--num_fewshot",
    type=int,
    default=None,
    help="Few-shot count; leave unset to use each task's default",
)
parser.add_argument(
    "--gpu_memory_utilization",
    type=float,
    default=0.9,
    help="GPU memory utilization for vLLM (0.0-1.0)",
)
parser.add_argument(
    "--tensor_parallel_size",
    type=int,
    default=1,
    help="Number of tensor parallel replicas for vLLM",
)
parser.add_argument(
    "--max_model_len",
    type=int,
    default=None,
    help="Maximum model length for vLLM",
)
parser.add_argument(
    "--quantization",
    type=str,
    default=None,
    help="Quantization method for vLLM (e.g., 'fp8', 'int4')",
)
parser.add_argument(
    "--trust_remote_code",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable or disable trust remote code",
)
args = parser.parse_args()

# Build model args for vLLM
model_args = args.model_args

# Add vLLM-specific parameters
if args.gpu_memory_utilization:
    model_args += f",gpu_memory_utilization={args.gpu_memory_utilization}"
if args.tensor_parallel_size:
    model_args += f",tensor_parallel_size={args.tensor_parallel_size}"
if args.max_model_len:
    model_args += f",max_model_len={args.max_model_len}"
if args.quantization:
    model_args += f",quantization={args.quantization}"
if args.trust_remote_code:
    model_args += ",trust_remote_code=True"

# Run evaluation
evaluate_kwargs = dict(
    model=args.model,
    model_args=model_args,
    tasks=args.tasks.split(","),
    batch_size=args.batch_size,
    limit=args.limit,
    device="cuda",
)
# Only override the task default if explicitly provided.
if args.num_fewshot is not None:
    evaluate_kwargs["num_fewshot"] = args.num_fewshot

# Validate CUDA availability and usage
if torch.cuda.is_available():
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CUDA is available, using GPU for evaluation"
    )
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GPU Device: {torch.cuda.get_device_name(0)}"
    )
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
else:
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING: CUDA not available, falling back to CPU"
    )
    evaluate_kwargs["device"] = "cpu"

# Verify model was loaded on GPU
initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

if __name__ == "__main__":
    results = lm_eval.simple_evaluate(**evaluate_kwargs)

    # After evaluation, check final memory usage
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    if torch.cuda.is_available() and final_memory > initial_memory:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GPU memory used: {final_memory / 1024**3:.1f} GB"
        )
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model appears to be using GPU"
        )
    elif "config" in results and "model" in results["config"]:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model config: {results['config']['model']}"
        )
    else:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Could not verify GPU usage"
        )

    # Print results
    # print("\nResults:")
    # print(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_args = re.sub(r"[^A-Za-z0-9_.-]", "_", model_args)
    outfile = f"lm_eval_results_{safe_model_args}_{args.tasks}_{timestamp}.json"

    def fallback(o):
        try:
            return str(o)
        except Exception:
            return "<unserializable_object>"

    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=fallback)

    print(f"\nSaved results to {outfile}")
