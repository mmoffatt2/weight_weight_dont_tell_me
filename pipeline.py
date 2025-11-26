# pipeline.py

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run(cmd, desc):
    print(f"\n🚀 {desc}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": "."})


def main():
    parser = argparse.ArgumentParser("MoE Usage-Aware Pipeline")

    # ---- Core
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="runs")

    # ---- Tracing
    parser.add_argument("--trace_dataset", type=str, default="wikitext2")
    parser.add_argument("--trace_nsamples", type=int, default=64)
    parser.add_argument("--trace_seqlen", type=int, default=2048)
    parser.add_argument("--trace_batch_size", type=int, default=1)
    parser.add_argument("--trace_config", type=str, default="configs/moe_model_metadata.json")

    # ---- Quant
    parser.add_argument("--quant_dataset", type=str, default="wikitext2")
    parser.add_argument("--quant_nsamples", type=int, default=256)
    parser.add_argument("--quant_seqlen", type=int, default=2048)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--bit_config", type=str, default="configs/bit_assign.yaml")

    # ---- Eval (lm_eval)
    parser.add_argument("--eval_tasks", type=str, default="wikitext")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--num_fewshot", type=int, default=0)

    # ---- Control flags
    parser.add_argument("--skip_trace", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip_quant", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip_eval", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    model_short = args.model_name.split("/")[-1]
    run_dir = Path(args.output_root) / model_short
    run_dir.mkdir(parents=True, exist_ok=True)

    expert_counts = run_dir / f"{model_short}_expert_counts.pt"
    quant_out = run_dir / "quantized"

    # -----------------------------------------------------
    # 1. Tracing (moe_tracing.py)
    # -----------------------------------------------------
    if not args.skip_trace:
        run(
            [
                sys.executable,
                "tracing/moe_tracing.py",
                "--model_name", args.model_name,
                "--config_path", args.trace_config,
                "--dataset", args.trace_dataset,
                "--seqlen", str(args.trace_seqlen),
                "--nsamples", str(args.trace_nsamples),
                "--batch_size", str(args.trace_batch_size),
                "--save_dir", str(run_dir),
            ],
            "Tracing MoE routing (moe_tracing.py)",
        )

    # -----------------------------------------------------
    # 2. Quantization
    # -----------------------------------------------------
    if not args.skip_quant:
        run(
            [
                sys.executable,
                "quantization/usage_aware_quantization.py",
                "--model_name", args.model_name,
                "--expert_counts", str(expert_counts),
                "--output_dir", str(quant_out),
                "--dataset", args.quant_dataset,
                "--seqlen", str(args.quant_seqlen),
                "--nsamples", str(args.quant_nsamples),
                "--group_size", str(args.group_size),
                "--bit_config", args.bit_config,
            ],
            "Running usage-aware quantization",
        )

    # -----------------------------------------------------
    # 3. Evaluation (lm_eval)
    # -----------------------------------------------------
    if not args.skip_eval:
        model_args = f"pretrained={quant_out},trust_remote_code=True"
        run(
            [
                sys.executable,
                "eval/run_lm_eval.py",
                "--model", "hf",
                "--model_args", model_args,
                "--tasks", args.eval_tasks,
                "--batch_size", str(args.eval_batch_size),
                "--num_fewshot", str(args.num_fewshot),
                "--load_in_4bit", "False",
                "--load_in_8bit", "False",
            ]
            + (["--limit", str(args.eval_limit)] if args.eval_limit else []),
            "Running lm_eval",
        )

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
