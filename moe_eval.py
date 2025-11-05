#!/usr/bin/env python3
"""
Minimal lm-evaluation-harness wrapper for quick MoE experiments.

Example:
    python moe_eval.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
        --dataset gsm8k --limit 128 --num_fewshot 0
"""

import argparse
import json
import os
from datetime import datetime

from transformers import set_seed

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import numpy as np


TASK_ALIASES = {
    "gsm8k": ["gsm8k_cot"],
    "wikitext2": ["wikitext"],
    "wmt14": ["flores_101.de-en"],
    "humaneval": ["humaneval"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lm_eval on a Hugging Face model.")
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face repo id (e.g. mistralai/Mixtral-8x7B-Instruct-v0.1)",
    )
    parser.add_argument("--revision", help="Optional git revision for the model.")
    parser.add_argument(
        "--tasks", help="Comma-separated lm_eval tasks (e.g. gsm8k_cot,truthfulqa_mc)."
    )
    parser.add_argument(
        "--dataset",
        choices=TASK_ALIASES.keys(),
        help="Shortcut alias mapped to lm_eval task ids.",
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=0, help="Few-shot examples per task."
    )
    parser.add_argument(
        "--limit", type=int, help="Max number of eval samples per task."
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for evaluation."
    )
    parser.add_argument(
        "--dtype", help="Optional model dtype hint (float16, bfloat16, float32)."
    )
    parser.add_argument(
        "--device_map",
        help="Optional device map to pass to transformers (e.g. auto, balanced).",
    )
    parser.add_argument(
        "--load_in_8bit", action="store_true", help="Load model weights in 8-bit."
    )
    parser.add_argument(
        "--load_in_4bit", action="store_true", help="Load model weights in 4-bit."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow custom code from the model repo.",
    )
    parser.add_argument(
        "--output",
        help="Path for metrics JSON. Defaults to results/<model>_<timestamp>.json",
    )
    return parser.parse_args()


def resolve_tasks(args: argparse.Namespace) -> list[str]:
    if args.tasks:
        return [task.strip() for task in args.tasks.split(",") if task.strip()]
    if args.dataset:
        return TASK_ALIASES[args.dataset]
    raise SystemExit("No tasks provided. Use --tasks or --dataset.")


def build_model(args: argparse.Namespace) -> HFLM:
    model_kwargs = {
        "pretrained": args.model,
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "device_map": args.device_map,
        "load_in_8bit": args.load_in_8bit,
        "load_in_4bit": args.load_in_4bit,
    }
    if args.dtype:
        model_kwargs["dtype"] = args.dtype
    return HFLM(**model_kwargs)


def summarize(results: dict) -> list[str]:
    lines = []
    for task, metrics in results.get("results", {}).items():
        bits = []
        for name, value in metrics.items():
            if isinstance(value, dict) and "mean" in value:
                stderr = value.get("stderr")
                if stderr is not None:
                    bits.append(f"{name}={value['mean']:.4f}±{stderr:.4f}")
                else:
                    bits.append(f"{name}={value['mean']:.4f}")
            elif isinstance(value, (int, float)):
                bits.append(f"{name}={value:.4f}")
        lines.append(f"{task}: {', '.join(bits) if bits else 'no metrics reported'}")
    return lines

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tasks = resolve_tasks(args)
    print(f"🧪 Evaluating {args.model} on: {', '.join(tasks)}")

    model = build_model(args)

    eval_kwargs = {
        "tasks": tasks,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "random_seed": args.seed,
        "fewshot_random_seed": args.seed,
    }

    results = evaluator.simple_evaluate(model=model, **eval_kwargs)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_stub = args.model.split("/")[-1]
    output_path = args.output or os.path.join(
        "results", f"{model_stub}_lm_eval_{timestamp}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print(f"✅ Saved metrics to {output_path}")
    for line in summarize(results):
        print(f"   • {line}")


if __name__ == "__main__":
    main()
