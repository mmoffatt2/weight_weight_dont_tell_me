# -*- coding: utf-8 -*-
"""
Minimal MoE Router Tracing Utility
-----------------------------------
Author: Michael Moffatt
Date: 2025-01-18

Supports models exposing router logits via output metadata or internal hooks.
"""

import os
import json
import argparse
from collections import defaultdict

import torch
from datasets_loader import get_dataset_samples, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

set_seed(42)


def attach_router_hooks(model):
    """Attach hooks to router layers (e.g., mlp.gate or MoEGate)."""
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
            print(f"✅ Hook registered on {name}")

    return router_records


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

    print(f"✅ Model: {args.model_name}")
    print(f"   Layers={num_layers}, Experts={num_experts}, Top-k={top_k}")

    is_deepseek = "deepseek" in args.model_name.lower()

    # Load tokenizer
    model_path = args.model_path if args.model_path else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = get_dataset_samples(args.dataset, tokenizer, args.seqlen, args.nsamples)
    data_loader = DataLoader(
        Dataset.from_list(dataset),
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        shuffle=True,
    )

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    # Disable caching
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"

    expert_counts = torch.zeros(num_layers, num_experts, dtype=torch.long)
    per_token_records = []

    # Choose routing capture method
    if router_key:
        print("📦 Using direct router output from model forward()")
        use_hooks = False
        router_records = None
    else:
        print("🪝 Using hooks to capture routing internally")
        use_hooks = True
        router_records = attach_router_hooks(model)

    # Inference loop
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Tracing")):
        batch = {k: v.to(device) for k, v in batch.items()}
        if "labels" in batch:
            batch.pop("labels")

        local_forward_kwargs = dict(forward_kwargs)
        local_forward_kwargs.setdefault("use_cache", False)

        with torch.no_grad():
            outputs = model(**batch, **local_forward_kwargs)

        # Direct router output
        if not use_hooks:
            router_data = getattr(outputs, router_key)

            if isinstance(router_data, (list, tuple)):
                router_data = torch.stack(router_data)

            if outputs_logits:
                probs = torch.softmax(router_data, dim=-1)
            else:
                probs = router_data

            topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)

            for layer_idx in range(num_layers):
                layer_experts = topk_indices[layer_idx].reshape(-1)
                unique, counts = torch.unique(layer_experts.cpu(), return_counts=True)
                expert_counts[layer_idx, unique] += counts

                per_token_records.append({
                    "batch_idx": batch_idx,
                    "layer_name": f"layer_{layer_idx}",
                    "topk_indices": topk_indices[layer_idx].cpu().tolist(),
                    "topk_values": topk_values[layer_idx].cpu().tolist(),
                })

        # Hook-based capture
        else:
            for layer_idx, (layer_name, tensors) in enumerate(router_records.items()):
                if is_deepseek:
                    layer_idx += 1

                if not tensors:
                    continue

                router_tensor = tensors[-1]
                if router_tensor.dtype in (torch.float16, torch.float32, torch.bfloat16):
                    probs = torch.softmax(router_tensor, dim=-1)
                    topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)
                else:
                    topk_indices = router_tensor
                    topk_values = None

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

            for k in router_records.keys():
                router_records[k].clear()

    # Save outputs
    base_name = args.model_name.split('/')[-1]
    torch.save(expert_counts, os.path.join(args.save_dir, f"{base_name}_expert_counts.pt"))
    with open(os.path.join(args.save_dir, f"{base_name}_token_trace.json"), "w") as f:
        json.dump(per_token_records, f)

    print(f"\n✅ Saved to {args.save_dir}")
    print(f"   ├─ {base_name}_expert_counts.pt")
    print(f"   └─ {base_name}_token_trace.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal MoE Router Tracing Utility")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--model_path", type=str, default=None, help="Local model path (optional)")
    parser.add_argument("--config_path", type=str, default="configs/moe_model_metadata.json")
    parser.add_argument("--dataset", type=str, default="wikitext2")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--nsamples", type=int, default=2048)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)

    args = parser.parse_args()
    dump_routing_trace(args)
