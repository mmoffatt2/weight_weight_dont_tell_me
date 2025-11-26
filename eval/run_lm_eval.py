#!/usr/bin/env python3
"""
Basic script to run lm_eval.
"""

import argparse
import json
from datetime import datetime
import lm_eval
import re

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="hf")
parser.add_argument("--model_args", type=str, required=True)
parser.add_argument("--tasks", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument(
    "--num_fewshot",
    type=int,
    default=None,
    help="Few-shot count; leave unset to use each task's default",
)
parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=False, help="Enable or disable 4-bit quantization")
parser.add_argument("--load_in_8bit", action=argparse.BooleanOptionalAction, default=False, help="Enable or disable 8-bit quantization")
parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False, help="Enable or disable trust remote code")
args = parser.parse_args()

# Build model args with quantization if specified
model_args = args.model_args
if args.load_in_4bit:
    model_args += ",load_in_4bit=True"
if args.load_in_8bit:
    model_args += ",load_in_8bit=True"
if args.trust_remote_code:
    model_args += ",trust_remote_code=True"

# Run evaluation
evaluate_kwargs = dict(
    model=args.model,
    model_args=model_args,
    tasks=args.tasks.split(","),
    batch_size=args.batch_size,
    limit=args.limit,
)
# Only override the task default if explicitly provided.
if args.num_fewshot is not None:
    evaluate_kwargs["num_fewshot"] = args.num_fewshot

results = lm_eval.simple_evaluate(**evaluate_kwargs)

# Print results
# print("\nResults:")
# print(results)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
safe_model_args = re.sub(r'[^A-Za-z0-9_.-]', '_', model_args)
outfile = f"lm_eval_results_{safe_model_args}_{args.tasks}_{timestamp}.json"

def fallback(o):
    try:
        return str(o)
    except Exception:
        return "<unserializable_object>"

with open(outfile, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=fallback)

print(f"\nSaved results to {outfile}")
