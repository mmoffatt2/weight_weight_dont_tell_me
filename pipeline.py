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
    parser.add_argument("--dataset", type=str, default="wikitext2")

    # ---- Tracing
    parser.add_argument("--trace_dataset", type=str, default=None, help="Falls back to --dataset if not set")
    parser.add_argument("--trace_nsamples", type=int, default=64)
    parser.add_argument("--trace_seqlen", type=int, default=2048)
    parser.add_argument("--trace_batch_size", type=int, default=1)
    parser.add_argument("--trace_config", type=str, default="configs/moe_model_metadata.json")

    # ---- Pruning and Quantization
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["bottom_k", "top_k", "random_k"],
        default=None,
        help="Whether we use the bottom_k, top_k, or random_k experts when pruning/quantizing" \
        "bottom_k is the least used experts, top_k is the most used experts",
    )
    parser.add_argument("--k", type=int, default=None, help="Number of experts to prune/quantize")

    # ---- Pruning
    parser.add_argument("--prune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prune_dataset", type=str, default=None, help="Falls back to --dataset if not set")
    # parser.add_argument(
    #     "--prune_strategy",
    #     type=str,
    #     choices=["bottom_k", "top_k", "random_k"],
    #     default=None,
    #     help="Whether we use the bottom_k, top_k, or random_k experts when pruning. Falls back to --strategy if not set",
    # )
    # parser.add_argument("--prune_k", type=int, default=None, help="Override k for pruning (if different from quantization). Falls back to --k if not set")

    # ---- Quantization
    parser.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quant_dataset", type=str, default=None, help="Falls back to --dataset if not set")
    # parser.add_argument(
    #     "--quant_strategy",
    #     type=str,
    #     choices=["bottom_k", "top_k", "random_k"],
    #     default=None,
    #     help="Whether we use the bottom_k, top_k, or random_k experts when quantizing. Falls back to --strategy if not set",
    # )
    # parser.add_argument("--quant_k", type=int, default=None, help="Override k for quantization (if different from pruning). Falls back to --k if not set")
    parser.add_argument("--quant_nsamples", type=int, default=256)
    parser.add_argument("--quant_seqlen", type=int, default=2048)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument(
        "--low-bits",
        type=int,
        default=8,
        help="Number of bits for low-precision experts",
    )
    parser.add_argument(
        "--high-bits",
        type=int,
        default=16,
        help="Number of bits for high-precision experts (default full precision)",
    )

    # ---- Eval (lm_eval)
    parser.add_argument("--eval_tasks", type=str, default="wikitext")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--num_fewshot", type=int, default=0)

    # ---- Control flags
    parser.add_argument(
        "--skip_trace", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--skip_eval", action=argparse.BooleanOptionalAction, default=False
    )

    args = parser.parse_args()

    model_short = args.model_name.split("/")[-1]

    # if args.prune:
    #     if args.k is None and args.prune_k is None:
    #         raise ValueError("You must specify the k number of experts to prune")
    #     if args.strategy is None and args.prune_strategy is None:
    #         raise ValueError("You must specify a strategy for pruning (bottom_k, top_k, random_k)")
    # if args.quantize:
    #     if args.k is None and args.quant_k is None:
    #         raise ValueError("You must specify the k number of experts to quantize")
    #     if args.strategy is None and args.quant_strategy is None:
    #         raise ValueError("You must specify a strategy for quantization (bottom_k, top_k, random_k)")
    if args.k is None:
        raise ValueError("You must specify the k number of experts to prune/quantize")
    if args.strategy is None:
        raise ValueError("You must specify a strategy for pruning/quantization (bottom_k, top_k, random_k)")

    # Build run directory name with dataset and expert info
    # TODO: if specific dataset args are set, might want to include those in the name
    run_dir_name = f"{model_short}_{args.dataset}_{args.k}experts_{args.strategy}"

    run_dir = Path(args.output_root) / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    expert_counts = run_dir / f"{model_short}_expert_counts.pt"
    if args.prune:
        prune_out = run_dir / "pruned"
    if args.quantize:
        quant_out = run_dir / "quantized"

    # -----------------------------------------------------
    # 1. Tracing (moe_tracing.py)
    # -----------------------------------------------------
    if not args.skip_trace:
        # Use trace_dataset if provided, otherwise fall back to dataset
        trace_dataset = args.dataset if args.trace_dataset is None else args.trace_dataset
        run(
            [
                sys.executable,
                "tracing/moe_tracing.py",
                "--model_name",
                args.model_name,
                "--config_path",
                args.trace_config,
                "--dataset",
                trace_dataset,
                "--seqlen",
                str(args.trace_seqlen),
                "--nsamples",
                str(args.trace_nsamples),
                "--batch_size",
                str(args.trace_batch_size),
                "--save_dir",
                str(run_dir),
            ],
            "Tracing MoE routing (moe_tracing.py)",
        )

    # -----------------------------------------------------
    # 2. Pruning
    # -----------------------------------------------------
    if args.prune:
        # Use prune_dataset if provided, otherwise fall back to dataset
        prune_dataset = args.dataset if args.prune_dataset is None else args.prune_dataset

        prune_cmd = [
            sys.executable,
            "pruning/usage_aware_pruning.py",
            "--model_name",
            args.model_name,
            "--dataset",
            prune_dataset,
            "--strategy",
            args.strategy,
            "--k",
            str(args.k),
            "--output_dir",
            str(prune_out),
        ]

        # TODO: not sure what this code is supposed to be doing
        # # Add expert_counts if available (for bottom_k and top_k strategies)
        # if args.prune_strategy in ["bottom_k", "top_k"] and expert_counts.exists():
        #     prune_cmd.extend(["--expert_counts", str(expert_counts)])
        # elif args.prune_strategy in ["bottom_k", "top_k"]:
        #     prune_cmd.extend(["--dataset", prune_dataset])

        run(prune_cmd, f"Running {args.strategy} pruning")

    # -----------------------------------------------------
    # 3. Quantization
    # -----------------------------------------------------
    if args.quantize:
        # Use quant_dataset if provided, otherwise fall back to dataset
        quant_dataset = args.dataset if args.quant_dataset is None else args.quant_dataset
        quant_cmd = [
            sys.executable,
            "quantization/usage_aware_quantization.py",
            "--model_name",
            args.model_name,
            "--expert_counts",
            str(expert_counts),
            "--output_dir",
            str(quant_out),
            "--dataset",
            quant_dataset,
            "--seqlen",
            str(args.quant_seqlen),
            "--nsamples",
            str(args.quant_nsamples),
            "--group_size",
            str(args.group_size),
            "--k",
            str(args.k),
            "--low-bits", 
            str(args.low_bits),
            "--high-bits", 
            str(args.high_bits),
            "--strategy", 
            args.strategy,
        ]

        run(quant_cmd, f"Running {args.strategy} quantization")

    # -----------------------------------------------------
    # 4. Evaluation (lm_eval)
    # -----------------------------------------------------
    if not args.skip_eval:
        model_paths = []

        if args.prune:
            # Pruning (with or without quantization)
            model_paths.append(str(prune_out))
        if args.quantize:
            # Quantization only
            model_paths.append(str(quant_out))

        if not model_paths:
            print("No models found for evaluation (neither pruning nor quantization was run). Skipping eval.")
            return
        for model_path in model_paths:
            eval_cmd = [
                sys.executable,
                "eval/eval_usage_aware_quantized.py",
                "--model_name",
                args.model_name,
                "--model_path",
                model_path,
                "--tasks",
                args.eval_tasks,
                "--batch_size",
                str(args.eval_batch_size),
            ]
            # Only add num_fewshot if explicitly set
            if args.num_fewshot is not None:
                eval_cmd.extend(["--num_fewshot", str(args.num_fewshot)])
            eval_cmd.extend(["--output_dir", str(run_dir)])

            run(eval_cmd, "Running usage-aware evaluation")

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
