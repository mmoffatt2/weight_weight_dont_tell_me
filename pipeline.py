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
    parser.add_argument("--top-k", type=int, default=None, 
                        help="Number of lowest-usage experts to quantize (overrides k in bit_assign.yaml)")
    parser.add_argument("--low-bits", type=int, default=None,
                        help="Number of bits for low-precision experts (overrides low_bits in bit_assign.yaml)")

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
    
    # Load bit config to get k value
    import yaml
    with open(args.bit_config, 'r') as f:
        bit_config = yaml.safe_load(f)
    
    # Determine k: use --top-k if provided, otherwise use k from config
    if args.top_k is not None:
        k_experts_to_quant = args.top_k
    else:
        k_experts_to_quant = bit_config.get('global_bottom_k', {}).get('k', 0)
    
    # Build run directory name with dataset and number of quantized experts
    run_dir_name = f"{model_short}_{args.trace_dataset}_{k_experts_to_quant}experts"
    
    run_dir = Path(args.output_root) / run_dir_name
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
        quant_cmd = [
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
        ]
        
        # Add --k override if provided
        if args.top_k is not None:
            quant_cmd.extend(["--k", str(args.top_k)])
        
        # Add --low-bits override if provided
        if args.low_bits is not None:
            quant_cmd.extend(["--low-bits", str(args.low_bits)])
        
        run(quant_cmd, "Running usage-aware quantization")

    # -----------------------------------------------------
    # 3. Evaluation (lm_eval)
    # -----------------------------------------------------
    if not args.skip_eval:
        eval_cmd = [
            sys.executable,
            "eval/eval_usage_aware_quantized.py",
            "--model_name", args.model_name,
            "--quant_model_path", str(quant_out),
            "--tasks", args.eval_tasks,
            "--batch_size", str(args.eval_batch_size),
        ]
        # Only add num_fewshot if explicitly set
        if args.num_fewshot is not None:
            eval_cmd.extend(["--num_fewshot", str(args.num_fewshot)])
        eval_cmd.extend(["--output_dir", str(run_dir)])
        
        run(eval_cmd, "Running usage-aware evaluation")

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
