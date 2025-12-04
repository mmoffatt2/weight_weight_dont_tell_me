# -*- coding: utf-8 -*-
"""
Universal MoE Router Tracing Utility (Metadata + Hook Fallback)
---------------------------------------------------------------
Author: Michael Moffatt
Date: 2025-10-13

Supports both:
 - Native models exposing router logits (e.g., Mixtral)
 - Models requiring internal hook capture (e.g., DeepSeek-MoE)
 - Quantized models using auto_gptq
"""

import os
import sys
import random
import json
import argparse
from collections import defaultdict

# Add MoE-Quantization to path for auto_gptq support
sys.path.insert(0, "./external/MoE-Quantization")

import torch
from utils.datasets_loader import get_dataset_samples, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

set_seed(42)


# ------------------------------------------------------------------------------
# Hook setup
# ------------------------------------------------------------------------------
def attach_router_hooks(model):
    """Attach hooks to likely router layers (e.g., MoEGate or DeepseekMoE.mlp.gate)."""
    router_records = defaultdict(list)

    def capture_router(name):
        def hook_fn(module, inputs, outputs):
            if torch.is_tensor(outputs):
                router_records[name].append(outputs.detach().cpu())
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
                router_records[name].append(outputs[0].detach().cpu())
        return hook_fn

    for name, module in model.named_modules():
        lname = name.lower()
        if lname.endswith("mlp.gate") or "moegate" in lname or "router" in lname:
            module.register_forward_hook(capture_router(name))
            print(f"✅ Hook registered on {name} ({module.__class__.__name__})")

    return router_records


# ------------------------------------------------------------------------------
# Main tracing function
# ------------------------------------------------------------------------------
def dump_routing_trace(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # Load metadata
    with open(args.config_path) as f:
        metadata = json.load(f)

    if args.model_name not in metadata:
        raise ValueError(f"Model {args.model_name} not found in {args.config_path}")

    model_info = metadata[args.model_name]
    router_key = model_info.get("router_key", None)
    num_layers = model_info["num_hidden_layers"]
    num_experts = model_info["num_routed_experts"]
    outputs_logits = model_info["outputs_logits"]
    top_k = model_info["num_experts_per_token"]
    forward_kwargs = model_info.get("forward_kwargs", {})

    print(f"✅ Using metadata for {args.model_name}")
    print(f"   Layers={num_layers}, Experts={num_experts}, Top-k={top_k}, Router key='{router_key}'")

    # Detect DeepSeek MoE models (all DeepSeek MoE family share the same pattern)
    is_deepseek = "deepseek" in args.model_name.lower()
    
    # --------------------------------------------------------------------------
    # Detect if quantized model and determine tokenizer source
    # --------------------------------------------------------------------------
    # Use model_path if provided (for local models), otherwise use model_name (HuggingFace)
    model_load_path = args.model_path if hasattr(args, 'model_path') and args.model_path else args.model_name
    
    # Detect if this is a quantized model by checking for quantize_config.json
    is_quantized = False
    if os.path.isdir(model_load_path):
        quantize_config_path = os.path.join(model_load_path, "quantize_config.json")
        if os.path.exists(quantize_config_path):
            is_quantized = True
            print("🔍 Detected quantized model (found quantize_config.json)")
    
    # For quantized models, load tokenizer from the base model name (not the quantized path)
    # since the quantized folder may not have a proper config.json
    tokenizer_source = args.model_name if is_quantized else model_load_path
    
    # Check if tokenizer files exist in the model path
    if os.path.isdir(model_load_path):
        has_tokenizer_files = any(
            os.path.exists(os.path.join(model_load_path, t))
            for t in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
        )
        if has_tokenizer_files and not is_quantized:
            tokenizer_source = model_load_path
    
    # --------------------------------------------------------------------------
    # Tokenizer + dataset
    # --------------------------------------------------------------------------
    print(f"Loading tokenizer from: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = get_dataset_samples(args.dataset, tokenizer, args.seqlen, args.nsamples)
    data_loader = DataLoader(
        Dataset.from_list(dataset),
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        shuffle=True,
    )

    # --------------------------------------------------------------------------
    # Model loading (patched to avoid DynamicCache / new-attn paths)
    # --------------------------------------------------------------------------
    print(f"Loading model from: {model_load_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if is_quantized:
        # Load quantized model using auto_gptq
        try:
            from auto_gptq import AutoGPTQForCausalLM_mixed_precision
        except ImportError:
            print("❌ auto_gptq not found. Please install it or ensure external/MoE-Quantization is available.")
            raise
        
        # Find the model basename (weight file prefix)
        quantized_model_file_base_name = None
        if os.path.isdir(model_load_path):
            try:
                dir_files = os.listdir(model_load_path)
                safetensors = [f for f in dir_files if f.endswith('.safetensors')]
                other_bins = [f for f in dir_files if f.endswith('.pt') or f.endswith('.bin') or f.endswith('.pth')]
                if safetensors:
                    quantized_model_file_base_name = os.path.splitext(safetensors[0])[0]
                elif other_bins:
                    quantized_model_file_base_name = os.path.splitext(other_bins[0])[0]
            except Exception:
                pass
        
        if not quantized_model_file_base_name:
            quantized_model_file_base_name = os.path.basename(os.path.normpath(model_load_path))
        
        print(f"   Loading quantized model with basename: {quantized_model_file_base_name}")
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        model = AutoGPTQForCausalLM_mixed_precision.from_quantized(
            model_load_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            model_basename=quantized_model_file_base_name,
            use_safetensors=True,
            trust_remote_code=True,
            inject_fused_mlp=False,
            inject_fused_attention=False,
        )
        print("✓ Quantized model loaded successfully")
    else:
        # Load standard (non-quantized) model
        # IMPORTANT: do NOT use device_map="auto" to avoid DynamicCache / sdpa corner cases
        model = AutoModelForCausalLM.from_pretrained(
            model_load_path,  # Load from local path if provided, otherwise HuggingFace
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        model.to(device)
        print("✓ Standard model loaded successfully")
    
    model.eval()

    # Disable caching to avoid HF DynamicCache in newer transformers
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Force eager attention implementation (avoids some SDPA/DynamicCache paths)
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"

    expert_counts = torch.zeros(num_layers, num_experts, dtype=torch.long)
    per_token_records = []

    # --------------------------------------------------------------------------
    # Choose routing capture method
    # --------------------------------------------------------------------------
    if router_key:
        print("📦 Using direct router output from model forward()")
        use_hooks = False
        router_records = None
    else:
        print("🪝 No router_key found — using hooks to capture routing internally")
        use_hooks = True
        router_records = attach_router_hooks(model)

    # --------------------------------------------------------------------------
    # Inference loop
    # --------------------------------------------------------------------------
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Dumping routing trace")):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}            # ### NEW
        if "labels" in batch:
            batch.pop("labels")

        # Ensure we don't accidentally override our no-cache setting via metadata
        local_forward_kwargs = dict(forward_kwargs)                    # ### NEW
        local_forward_kwargs.setdefault("use_cache", False)            # ### NEW

        with torch.no_grad():
            outputs = model(**batch, **local_forward_kwargs)

        # --- Case 1: Direct router output from model outputs ---
        if not use_hooks:
            router_data = getattr(outputs, router_key)

            # Some models return a list/tuple over layers
            if isinstance(router_data, (list, tuple)):
                router_data = torch.stack(router_data)

            # router_data shape: [num_layers, batch, seq, num_experts] (usually)
            if outputs_logits:
                probs = torch.softmax(router_data, dim=-1)
            else:
                probs = router_data

            topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)

            # Aggregate expert counts per layer
            for layer_idx in range(num_layers):
                # Flatten over batch/seq
                layer_experts = topk_indices[layer_idx].reshape(-1)
                unique, counts = torch.unique(layer_experts.cpu(), return_counts=True)
                expert_counts[layer_idx, unique] += counts

                per_token_records.append({
                    "batch_idx": batch_idx,
                    "layer_name": f"layer_{layer_idx}",
                    "topk_indices": topk_indices[layer_idx].cpu().tolist(),
                    "topk_values": topk_values[layer_idx].cpu().tolist(),
                })

        # --- Case 2: Hook-based capture ---
        else:
            # router_records: {layer_name: [tensor_batch_0, tensor_batch_1, ...]}
            # We only use the *last* tensor per layer for this batch.
            for layer_idx, (layer_name, tensors) in enumerate(router_records.items()):
                if is_deepseek:
                    layer_idx += 1

                if not tensors:
                    continue

                router_tensor = tensors[-1]  # [batch, seq, num_experts] or [batch, seq]  # last recorded batch
                # Handle both logits and integer expert IDs
                if router_tensor.dtype in (torch.float16, torch.float32, torch.bfloat16):
                    probs = torch.softmax(router_tensor, dim=-1)
                    topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)
                else:
                    topk_indices = router_tensor
                    topk_values = None

                # Aggregate counts
                flat_experts = topk_indices.reshape(-1)
                unique, counts = torch.unique(flat_experts, return_counts=True)

                if layer_idx < num_layers:
                    expert_counts[layer_idx, unique] += counts

                per_token_records.append({
                    "batch_idx": batch_idx,
                    "layer_name": layer_name,
                    "topk_indices": topk_indices.cpu().tolist(),
                    "topk_values": topk_values.cpu().tolist() if topk_values is not None else [],
                })

            # (Optional but cleaner) clear records so next batch doesn't grow forever  # ### NEW
            for k in router_records.keys():                                           # ### NEW
                router_records[k].clear()                                             # ### NEW

    # --------------------------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------------------------
    # Determine base name for output files:
    # 1. Use model_name_override if explicitly provided
    # 2. If model_path is provided, use the parent folder name (for quantized subfolders)
    # 3. Otherwise, extract from model_name (HuggingFace ID)
    is_quantized = False
    if hasattr(args, 'model_name_override') and args.model_name_override:
        model_base_name = args.model_name_override
    elif hasattr(args, 'model_path') and args.model_path:
        # If model_path ends with 'quantized', use parent folder name and mark as quantized
        if args.model_path.endswith('quantized') or args.model_path.endswith('quantized/'):
            model_base_name = os.path.basename(os.path.dirname(args.model_path.rstrip('/')))
            is_quantized = True
        else:
            model_base_name = os.path.basename(args.model_path)
    else:
        model_base_name = args.model_name.split('/')[-1]
    
    # Add 'quantized_' prefix if loading from quantized folder
    if is_quantized:
        base_name = f"quantized_{model_base_name}_{args.dataset}"
    else:
        base_name = f"{model_base_name}"
    
    torch.save(expert_counts, os.path.join(args.save_dir, f"{base_name}_expert_counts.pt"))
    with open(os.path.join(args.save_dir, f"{base_name}_token_trace.json"), "w") as f:
        json.dump(per_token_records, f)

    print(f"\n✅ Saved routing data to {args.save_dir}")
    print(f"   ├─ {base_name}_expert_counts.pt  (tensor [{num_layers} × {num_experts}])")
    print(f"   └─ {base_name}_token_trace.json   (per-token Top-{top_k} expert indices & gate weights)")


# ------------------------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal MoE Router Tracing Utility (metadata + hook fallback)")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model ID (must exist in metadata JSON)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to local model directory (e.g., quantized folder). If not provided, loads from HuggingFace using model_name. Output filenames will use the parent folder name.")
    parser.add_argument("--model_name_override", type=str, default=None, help="Override model name for output filenames. If not provided and model_path is set, uses parent folder name.")
    parser.add_argument("--config_path", type=str, default="configs/moe_model_metadata.json", help="Path to MoE model metadata JSON")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext2",
        help="Dataset name: wikitext2 | gsm8k | wmt14 | humaneval | ds1000 | swebench | agentbench",
    )
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save routing data")
    parser.add_argument("--nsamples", type=int, default=64, help="Number of random sequences to test")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length per sample")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")

    args = parser.parse_args()
    dump_routing_trace(args)
