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
    parser.add_argument(
        "--trace_config", type=str, default="configs/moe_model_metadata.json"
    )

    # ---- Quant
    parser.add_argument("--quant_dataset", type=str, default="wikitext2")
    parser.add_argument("--quant_nsamples", type=int, default=256)
    parser.add_argument("--quant_seqlen", type=int, default=2048)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--bit_config", type=str, default="configs/bit_assign.yaml")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of lowest-usage experts to quantize (overrides k in bit_assign.yaml)",
    )
    parser.add_argument(
        "--low-bits",
        type=int,
        default=None,
        help="Number of bits for low-precision experts (overrides low_bits in bit_assign.yaml)",
    )

    # ---- Eval (lm_eval)
    parser.add_argument("--eval_tasks", type=str, default="wikitext")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation (higher for vLLM)",
    )
    parser.add_argument("--eval_limit", type=int, default=None)
    parser.add_argument("--num_fewshot", type=int, default=0)

    # ---- Pruning
    parser.add_argument(
        "--prune_strategy",
        type=str,
        choices=["bottom_k", "top_k", "random"],
        default=None,
        help="Pruning strategy: bottom_k, top_k, or random",
    )
    parser.add_argument(
        "--prune_k",
        type=int,
        default=None,
        help="Number of experts to remove per layer",
    )
    parser.add_argument(
        "--prune_dataset",
        type=str,
        default=None,
        help="Dataset for usage-based pruning (falls back to --trace_dataset)",
    )
    parser.add_argument(
        "--reuse_trace_data",
        action="store_true",
        default=False,
        help="Reuse existing trace data for pruning (use with --skip_trace)",
    )

    # ---- Control flags
    parser.add_argument(
        "--skip_trace", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--skip_quant", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--skip_eval", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--skip_prune", action=argparse.BooleanOptionalAction, default=False
    )

    args = parser.parse_args()

    model_short = args.model_name.split("/")[-1]

    # Load bit config to get k value
    import yaml

    with open(args.bit_config, "r") as f:
        bit_config = yaml.safe_load(f)

    # Determine k: use --top-k if provided, otherwise use k from config
    if args.top_k is not None:
        k_experts_to_quant = args.top_k
    else:
        k_experts_to_quant = bit_config.get("global_bottom_k", {}).get("k", 0)

    # Determine if pruning is enabled
    pruning_enabled = (
        args.prune_strategy is not None
        and args.prune_k is not None
        and not args.skip_prune
    )

    # Build run directory name with dataset and expert info
    if pruning_enabled:
        run_dir_name = f"{model_short}_{args.trace_dataset}_{args.prune_k}experts_{args.prune_strategy}"
    else:
        run_dir_name = f"{model_short}_{args.trace_dataset}_{k_experts_to_quant}experts"

    run_dir = Path(args.output_root) / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # When skipping trace, reuse existing expert counts from baseline run
    if args.skip_trace and args.reuse_trace_data:
        baseline_dir = (
            Path(args.output_root)
            / f"{model_short}_{args.trace_dataset}_{k_experts_to_quant}experts"
        )
        expert_counts = baseline_dir / f"{model_short}_expert_counts.pt"
        if not expert_counts.exists():
            print(f"❌ Cannot find expert counts at {expert_counts}")
            print("   Run baseline evaluation first or remove --skip_trace")
            sys.exit(1)
        print(f"📊 Reusing existing trace data from {baseline_dir}")
    else:
        expert_counts = run_dir / f"{model_short}_expert_counts.pt"

    prune_out = run_dir / "pruned"
    quant_out = run_dir / "quantized"

    # -----------------------------------------------------
    # 1. Tracing (moe_tracing.py)
    # -----------------------------------------------------
    if not args.skip_trace:
        run(
            [
                sys.executable,
                "tracing/moe_tracing.py",
                "--model_name",
                args.model_name,
                "--config_path",
                args.trace_config,
                "--dataset",
                args.trace_dataset,
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
    # 2. Pruning (optional)
    # -----------------------------------------------------
    if pruning_enabled:
        # Use prune_dataset if provided, otherwise fall back to trace_dataset
        prune_dataset = args.prune_dataset or args.trace_dataset

        prune_cmd = [
            sys.executable,
            "pruning/usage_aware_pruning.py",
            "--model_name",
            args.model_name,
            "--strategy",
            args.prune_strategy,
            "--k",
            str(args.prune_k),
            "--output_dir",
            str(prune_out),
        ]

        # Add expert_counts if available (for bottom_k and top_k strategies)
        if args.prune_strategy in ["bottom_k", "top_k"] and expert_counts.exists():
            prune_cmd.extend(["--expert_counts", str(expert_counts)])
        elif args.prune_strategy in ["bottom_k", "top_k"]:
            prune_cmd.extend(["--dataset", prune_dataset])

        run(prune_cmd, f"Running {args.prune_strategy} pruning")

    # -----------------------------------------------------
    # 3. Quantization
    # -----------------------------------------------------
    if not args.skip_quant:
        # Determine which model to use for quantization
        model_for_quant = args.model_name
        if pruning_enabled:
            model_for_quant = str(prune_out)

        quant_cmd = [
            sys.executable,
            "quantization/usage_aware_quantization.py",
            "--model_name",
            model_for_quant,
            "--expert_counts",
            str(expert_counts),
            "--output_dir",
            str(quant_out),
            "--dataset",
            args.quant_dataset,
            "--seqlen",
            str(args.quant_seqlen),
            "--nsamples",
            str(args.quant_nsamples),
            "--group_size",
            str(args.group_size),
            "--bit_config",
            args.bit_config,
        ]

        # Add --k override if provided
        if args.top_k is not None:
            quant_cmd.extend(["--k", str(args.top_k)])

        # Add --low-bits override if provided
        if args.low_bits is not None:
            quant_cmd.extend(["--low-bits", str(args.low_bits)])

        run(quant_cmd, "Running usage-aware quantization")

    # -----------------------------------------------------
    # 4. Evaluation (lm_eval)
    # -----------------------------------------------------
    if not args.skip_eval:
        # Determine which model to use for evaluation
        model_for_eval = args.model_name
        quant_model_path_for_eval = str(quant_out)

        if pruning_enabled and not args.skip_quant:
            # Pruning + quantization
            model_for_eval = str(prune_out)  # Use pruned model as base
            quant_model_path_for_eval = str(quant_out)
        elif pruning_enabled and args.skip_quant:
            # Pruning only, no quantization
            model_for_eval = str(prune_out)
            quant_model_path_for_eval = str(prune_out)  # Use same path for consistency
        elif not pruning_enabled and not args.skip_quant:
            # Quantization only
            model_for_eval = args.model_name
            quant_model_path_for_eval = str(quant_out)
        # else: neither pruning nor quantization, use original model

        # Use vLLM for faster evaluation whenever possible
        if args.skip_quant or (not pruning_enabled and args.skip_quant):
            # For baseline and pruning-only cases, use vLLM for much faster evaluation
            eval_cmd = [
                sys.executable,
                "eval/run_lm_eval.py",
                "--model",
                "vllm",
                "--model_args",
                f"pretrained={model_for_eval},trust_remote_code=True,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=4096",
                "--tasks",
                args.eval_tasks,
                "--batch_size",
                str(args.eval_batch_size),
                "--trust_remote_code",
            ]
            # Only add num_fewshot if explicitly set
            if args.num_fewshot is not None:
                eval_cmd.extend(["--num_fewshot", str(args.num_fewshot)])
            if args.eval_limit is not None:
                eval_cmd.extend(["--limit", str(args.eval_limit)])

            run(eval_cmd, "Running vLLM evaluation (non-quantized)")
        else:
            # For quantized models, use the specialized evaluation script
            eval_cmd = [
                sys.executable,
                "eval/eval_usage_aware_quantized.py",
                "--model_name",
                model_for_eval,
                "--quant_model_path",
                quant_model_path_for_eval,
                "--tasks",
                args.eval_tasks,
                "--batch_size",
                str(args.eval_batch_size),
            ]
            # Only add num_fewshot if explicitly set
            if args.num_fewshot is not None:
                eval_cmd.extend(["--num_fewshot", str(args.num_fewshot)])
            if args.eval_limit is not None:
                eval_cmd.extend(["--limit", str(args.eval_limit)])
            eval_cmd.extend(["--output_dir", str(run_dir)])

            run(eval_cmd, "Running usage-aware evaluation")

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
