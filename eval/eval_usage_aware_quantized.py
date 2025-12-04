#!/usr/bin/env python3
"""
Evaluation script for usage-aware quantized models using auto_gptq.

This script properly loads mixed-precision quantized models that have dictionary-based
bit assignments and evaluates them using lm_eval.

Usage:
    python eval/eval_usage_aware_quantized.py \
        --model_name deepseek-ai/deepseek-moe-16b-chat \
        --quant_model_path ./moe_usage_quantized_model \
        --tasks winogrande,copa,openbookqa,hellaswag,piqa,mmlu \
        --batch_size 16
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List

# ---- Add MoE-Quantization to path BEFORE other imports
sys.path.insert(0, "./external/MoE-Quantization")

import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM_mixed_precision
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks


# Task configurations for evaluation
LM_EVAL_TASK_KWARGS_DICT = {
    "winogrande": {
        "task": "winogrande",
        "num_fewshot": 0,
        "batch_size": 128,
        "metric": "acc",
    },
    "copa": {
        "task": "copa",
        "num_fewshot": 0,
        "batch_size": 128,
        "metric": "acc",
    },
    "openbookqa": {
        "task": "openbookqa",
        "num_fewshot": 0,
        "batch_size": 128,
        "metric": "acc_norm",
    },
    "hellaswag": {
        "task": "hellaswag",
        "num_fewshot": 0,
        "batch_size": 128,
        "metric": "acc_norm",
    },
    "piqa": {
        "task": "piqa",
        "num_fewshot": 0,
        "batch_size": 128,
        "metric": "acc",
    },
    "mmlu": {
        "task": "mmlu",
        "num_fewshot": 5,
        "batch_size": 16,
        "metric": "acc",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate usage-aware quantized models with lm_eval"
    )

    # Model paths
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Original base model name (HuggingFace hub)",
    )
    parser.add_argument(
        "--quant_model_path",
        type=str,
        required=True,
        help="Path to quantized model directory",
    )

    # Evaluation configuration
    parser.add_argument(
        "--tasks",
        type=str,
        default="winogrande,copa,openbookqa,hellaswag,piqa,mmlu",
        help="Comma-separated list of tasks to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation (can be overridden per-task)",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (overrides task defaults if set)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Limit the number of examples per task (for quick tests)",
    )

    # Model loading options
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        default=False,
        help="Use fast tokenizer",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Allow remote code execution",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--skip_mmlu",
        action="store_true",
        default=False,
        help="Skip MMLU for faster testing",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load model and tokenizer
    # --------------------------------------------------------
    print(f"\n📥 Loading quantized model from {args.quant_model_path}")
    print(f"   Base model: {args.model_name}")

    # Try to find a safetensors / quant file inside the provided quantized model dir
    # AutoGPTQ expects a `model_basename` that matches the weight file prefix (without extension).
    # The original code assumed the last path segment was the basename which fails when
    # the safetensors file has a different name (e.g. `moe_usage_quantized.safetensors`).
    quantized_model_file_base_name = None
    if os.path.isdir(args.quant_model_path):
        try:
            dir_files = os.listdir(args.quant_model_path)
        except Exception:
            dir_files = []

        # look for common quant file extensions
        safetensors = [f for f in dir_files if f.endswith('.safetensors')]
        other_bins = [f for f in dir_files if f.endswith('.pt') or f.endswith('.bin') or f.endswith('.pth')]
        if safetensors:
            quantized_model_file_base_name = os.path.splitext(safetensors[0])[0]
        elif other_bins:
            quantized_model_file_base_name = os.path.splitext(other_bins[0])[0]
        else:
            # fallback to directory name
            quantized_model_file_base_name = os.path.basename(os.path.normpath(args.quant_model_path))
    else:
        # if a file path was given, strip extension
        quantized_model_file_base_name = os.path.splitext(os.path.basename(args.quant_model_path))[0]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Allow code-eval tasks like HumanEval
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

    # Prefer tokenizer shipped with the quantized folder if present (many quantization
    # pipelines save tokenizer files alongside the weights). Otherwise, fall back to
    # the original base model name on the Hub.
    tokenizer_source = args.model_name
    if os.path.isdir(args.quant_model_path):
        has_tokenizer_files = any(
            os.path.exists(os.path.join(args.quant_model_path, t))
            for t in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
        )
        if has_tokenizer_files:
            tokenizer_source = args.quant_model_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoGPTQForCausalLM_mixed_precision.from_quantized(
        args.quant_model_path,
        low_cpu_mem_usage=True,
        device_map="auto",
        model_basename=quantized_model_file_base_name,
        use_safetensors=True,
        trust_remote_code=args.trust_remote_code,
        inject_fused_mlp=False,
        inject_fused_attention=False,
    )

    print("✓ Model loaded successfully")

    # --------------------------------------------------------
    # 2. Parse task list (with alias expansion)
    # --------------------------------------------------------
    raw_task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    # Alias expansion for grouped shorthand names.
    # 'wmt' will expand to both directions for listed language pairs present in repo.
    # For consistency with tracing & quantization stages which use WMT14 de-en subset
    # we restrict the 'wmt' alias to a single German→English direction. The harness
    # does not provide a generated wmt14-de-en task yaml, so we fall back to the
    # closest available German→English pair: wmt16-de-en. If a custom wmt14-de-en
    # yaml is added later, replace the alias target accordingly.
    WMT_ALIAS_TARGET = "wmt16-de-en"
    expanded = []
    for t in raw_task_list:
        if t.lower() == "wmt":
            expanded.append(WMT_ALIAS_TARGET)
        else:
            expanded.append(t)
    task_list = expanded

    if args.skip_mmlu and "mmlu" in task_list:
        task_list.remove("mmlu")
        print("⊘ Skipping MMLU as requested")

    if not task_list:
        print("❌ No tasks specified!")
        return

    print(f"\n📊 Tasks to evaluate: {', '.join(task_list)}")
    if "wmt" in raw_task_list:
        print(f"   (Expanded 'wmt' alias into single task: {WMT_ALIAS_TARGET})")

    # --------------------------------------------------------
    # 3. Run evaluation
    # --------------------------------------------------------
    print(f"\n⚡ Starting evaluation...\n")

    # Initialize task registry once (loads YAMLs under external/MoE-Quantization/lm_eval/tasks)
    initialize_tasks(verbosity="INFO")

    all_metrics = {}
    results_per_task = {}

    for task_name in task_list:
        # Prepare per-task kwargs: use preset config when available, otherwise fall back to
        # command-line provided defaults so any lm-eval-harness task can be evaluated.
        if task_name in LM_EVAL_TASK_KWARGS_DICT:
            task_kwargs = LM_EVAL_TASK_KWARGS_DICT[task_name].copy()
        else:
            # default behavior for unknown tasks: no few-shot by default, use global batch size
            task_kwargs = {"task": task_name, "num_fewshot": 0, "batch_size": args.batch_size, "metric": None}

        # Override with command-line args if provided
        if args.num_fewshot is not None:
            task_kwargs["num_fewshot"] = args.num_fewshot
        if args.batch_size is not None:
            task_kwargs["batch_size"] = args.batch_size

        print(f"📋 Evaluating: {task_name}")
        print(f"   Fewshot: {task_kwargs['num_fewshot']}, Batch size: {task_kwargs['batch_size']}")

        try:
            lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=task_kwargs["batch_size"])

            # Use lm_eval evaluator to run the task. For unknown tasks we pass the task name directly
            # and rely on the harness to use sensible defaults.
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=task_name,
                num_fewshot=task_kwargs["num_fewshot"],
                batch_size=task_kwargs["batch_size"],
                log_samples=False,
                limit=args.limit,
            )

            # Store results. If we know the metric, extract it; otherwise save the whole task results.
            if task_name in results.get("results", {}):
                task_res = results["results"][task_name]
            else:
                # legacy: sometimes aliasing occurs, fall back to first key
                task_res = next(iter(results.get("results", {}).values()), {})

            if task_kwargs.get("metric"):
                metric = task_kwargs["metric"]
                metric_value = None
                for key, value in task_res.items():
                    if key.startswith(metric + ","):
                        metric_value = value
                        break
                if metric_value is not None:
                    all_metrics[f"{task_name}_{metric}"] = metric_value
                    results_per_task[task_name] = metric_value
                    print(f"   ✓ {metric}: {metric_value:.4f}")
                else:
                    # metric not found; save entire task result as fallback
                    results_per_task[task_name] = task_res
                    print(f"   ✓ Saved full results for {task_name}")
            else:
                # Unknown/default task: save the whole result dict so caller can inspect
                results_per_task[task_name] = task_res
                print(f"   ✓ Saved full results for {task_name}")

        except Exception as e:
            # Common cause: task name not registered. Provide a hint.
            msg = str(e)
            if "Missing task" in msg:
                print("   ⚠️ Task not found in registry. Double-check the YAML name under external/MoE-Quantization/lm_eval/tasks.")
                print("      If you added humaneval, use the exact task id from its YAML (e.g., 'humaneval' or 'humaneval_instruct').")
            print(f"   ❌ Error evaluating {task_name}: {e}")
            continue

    # --------------------------------------------------------
    # 4. Save results
    # --------------------------------------------------------
    print(f"\n📊 Evaluation complete!\n")
    print("Results:")
    print("-" * 50)
    for task_name, value in results_per_task.items():
        # Handle both scalar metrics and full dict results
        if isinstance(value, (int, float)):
            print(f"  {task_name:20s}: {value:.4f}")
        else:
            # Print a compact JSON for non-scalar results
            try:
                compact = json.dumps(value, ensure_ascii=False)[:400]
            except Exception:
                compact = str(value)
            print(f"  {task_name:20s}: {compact}")

            # Save all numeric metrics from dict results for this task
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        all_metrics[f"{task_name}_{k}"] = v
    print("-" * 50)

    if all_metrics:
        average = sum(all_metrics.values()) / len(all_metrics)
        all_metrics["average"] = average
        print(f"  {'average':20s}: {average:.4f}\n")
    else:
        print("  (No metrics collected)\n")

    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        args.output_dir,
        f"eval_result_{quantized_model_file_base_name}_{timestamp}.json",
    )

    output_data = {
        "model": args.model_name,
        "quantized_model_path": args.quant_model_path,
        "tasks": task_list,
        "metrics": all_metrics,
        "timestamp": timestamp,
    }

    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"💾 Results saved to: {results_file}\n")
    print("✅ Evaluation complete.")


if __name__ == "__main__":
    main()
