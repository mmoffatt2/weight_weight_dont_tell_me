# quantization/usage_aware_quantization.py

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer

# ---- Local imports
from quantization.usage_assign import assign_bits_from_usage
from quantization.calibration_inspector import inspect_calibration_dataset
from quantization.build_bitdict import build_deepseek_bitdict
from utils.datasets_loader import get_dataset_samples

# ---- Optional MoE-Quantization import (Option A)
sys.path.append("./external/MoE-Quantization")
from auto_gptq import AutoGPTQForCausalLM_mixed_precision, BaseQuantizeConfig_mixed_precision


def main():
    parser = argparse.ArgumentParser("MoE Usage-Aware Quantizer")

    # Model + files
    parser.add_argument("--model_name", type=str, required=True)
    # Expert counts file (...expert_counts.pt)
    parser.add_argument("--expert_counts", type=str, required=True)
    # Quantized model output directory
    parser.add_argument("--output_dir", type=str, default="./moe_usage_quantized_model")

    # Calibration dataset
    parser.add_argument("--dataset", type=str, default="wikitext2")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--nsamples", type=int, default=256)

    # Bit assignment
    parser.add_argument("--bit_config", type=str, default="configs/bit_assign.yaml")

    # Quant
    parser.add_argument("--group_size", type=int, default=128)

    # Debug
    parser.add_argument("--inspect-dataset", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable calibration dataset inspection")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load routing usage
    # --------------------------------------------------------
    print(f"\n📥 Loading expert counts: {args.expert_counts}")
    expert_counts = torch.load(args.expert_counts)

    num_layers, num_experts = expert_counts.shape
    print(f"Detected: {num_layers} layers x {num_experts} experts")

    # --------------------------------------------------------
    # 2. Assign bits
    # --------------------------------------------------------
    print("\n🧠 Assigning bits from usage...")
    bit_assignments = assign_bits_from_usage(
        expert_counts=expert_counts,
        config_path=args.bit_config,
    )

    # --------------------------------------------------------
    # 3. Build DeepSeek bitdict (NO shared / attn quant)
    # --------------------------------------------------------
    print("\n🧩 Building DeepSeek bitdict...")
    bitdict = build_deepseek_bitdict(
        bit_assignments=bit_assignments,
        num_layers=num_layers,
        num_experts=num_experts
    )

    print(f"Quantizing {len(bitdict)} tensors")

    # --------------------------------------------------------
    # 4. Load tokenizer + calibration data
    # --------------------------------------------------------
    print("\n📚 Loading tokenizer + dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_dataset = get_dataset_samples(
        name=args.dataset,
        tokenizer=tokenizer,
        seqlen=args.seqlen,
        nsamples=args.nsamples,
    )

    if args.inspect_dataset:
        inspect_calibration_dataset(quant_dataset, tokenizer)

    # --------------------------------------------------------
    # 5. Configure GPTQ
    # --------------------------------------------------------
    print("\n⚙️ Building GPTQ config...")
    quant_config = BaseQuantizeConfig_mixed_precision(
        bits=bitdict,
        group_size=args.group_size,
        desc_act=False,
        model_file_base_name="moe_usage_quantized",
    )

    # --------------------------------------------------------
    # 6. Load model
    # --------------------------------------------------------
    print("\n🧠 Loading model...")
    model = AutoGPTQForCausalLM_mixed_precision.from_pretrained(
        args.model_name,
        quant_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # --------------------------------------------------------
    # 7. Run quantization
    # --------------------------------------------------------
    print("\n⚡ Running calibration + quantization...")
    model.quantize(quant_dataset)

    # --------------------------------------------------------
    # 8. Save
    # --------------------------------------------------------
    print(f"\n💾 Saving quantized model to {args.output_dir}")
    model.save_quantized(args.output_dir)

    print("\n✅ Quantization complete.")


if __name__ == "__main__":
    main()
