#!/usr/bin/env python3
"""
Basic script to run lm_eval.
"""

import argparse

import lm_eval

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="hf")
parser.add_argument("--model_args", type=str, required=True)
parser.add_argument("--tasks", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--load_in_4bit", action="store_true")
parser.add_argument("--load_in_8bit", action="store_true")
args = parser.parse_args()

# Build model args with quantization if specified
model_args = args.model_args
if args.load_in_4bit:
    model_args += ",load_in_4bit=True"
if args.load_in_8bit:
    model_args += ",load_in_8bit=True"

# Run evaluation
results = lm_eval.simple_evaluate(
    model=args.model,
    model_args=model_args,
    tasks=args.tasks.split(","),
    batch_size=args.batch_size,
    limit=args.limit,
)

# Print results
print("\nResults:")
print(results)
