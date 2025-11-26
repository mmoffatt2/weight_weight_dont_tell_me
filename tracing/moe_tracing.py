# -*- coding: utf-8 -*-
"""
Universal MoE Router Tracing Utility (Metadata + Hook Fallback)
---------------------------------------------------------------
Author: Michael Moffatt
Date: 2025-10-13

Supports both:
 - Native models exposing router logits (e.g., Mixtral)
 - Models requiring internal hook capture (e.g., DeepSeek-MoE)
"""

import os
import random
import json
import argparse
from collections import defaultdict

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


# # ------------------------------------------------------------------------------
# # Dataset helper
# # ------------------------------------------------------------------------------
# def get_wikitext2(tokenizer, seqlen: int, nsamples: int, split: str = "train"):
#     """Randomly slice WikiText-2 sequences."""
#     data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train" if split == "train" else "test")
#     text = "".join([" \n" if s == "" else s for s in data["text"][:1000]])
#     enc = tokenizer(text, return_tensors="pt")
#     dataset = []
#     for _ in range(nsamples):
#         i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
#         j = i + seqlen
#         inp = enc.input_ids[:, i:j]
#         attn = torch.ones_like(inp)
#         dataset.append({"input_ids": inp, "attention_mask": attn})
#     return dataset


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
            elif isinstance(outputs, (tuple, list)) and torch.is_tensor(outputs[0]):
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

    # Load tokenizer, data, and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = get_dataset_samples(args.dataset, tokenizer, args.seqlen, args.nsamples)
    data_loader = DataLoader(
        Dataset.from_list(dataset),
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        shuffle=True,
    )

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    expert_counts = torch.zeros(num_layers, num_experts)
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
        batch = {k: v.cuda() for k, v in batch.items()}
        if "labels" in batch:
            batch.pop("labels")
        # if args.batch_size == 1:
        #     for k, v in batch.items():
        #         batch[k] = v.squeeze(0)

        with torch.no_grad():
            outputs = model(**batch, **forward_kwargs)

        # --- Case 1: Direct router output ---
        if not use_hooks:
            router_data = getattr(outputs, router_key)
            if isinstance(router_data, (list, tuple)):
                router_data = torch.stack(router_data)

            probs = torch.softmax(router_data, dim=-1) if outputs_logits else router_data

            topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)
            for layer_idx in range(num_layers):
                unique, counts = torch.unique(topk_indices[layer_idx], return_counts=True)
                expert_counts[layer_idx, unique.cpu()] += counts.cpu()

                per_token_records.append({
                    "batch_idx": batch_idx,
                    "layer_name": f"layer_{layer_idx}",
                    "topk_indices": topk_indices[layer_idx].cpu().tolist(),
                    "topk_values": topk_values[layer_idx].cpu().tolist(),
                })

        # --- Case 2: Hook-based capture ---
        else:
            # Flatten captured hooks for this batch
            for layer_idx, (layer_name, tensors) in enumerate(router_records.items()):
                if not tensors:
                    continue
                router_tensor = tensors[-1]  # last recorded batch

                # probs = torch.softmax(router_tensor, dim=-1)
                # topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)

                # Handle both logits and integer expert IDs
                if router_tensor.dtype in (torch.float16, torch.float32, torch.bfloat16):
                    probs = torch.softmax(router_tensor, dim=-1)
                    topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)
                else:
                    topk_indices = router_tensor
                    topk_values = None

                unique, counts = torch.unique(topk_indices, return_counts=True)
                if layer_idx < num_layers:
                    expert_counts[layer_idx, unique] += counts

                per_token_records.append({
                    "batch_idx": batch_idx,
                    "layer_name": layer_name,
                    "topk_indices": topk_indices.tolist(),
                    "topk_values": topk_values.cpu().tolist() if topk_values is not None else [],
                })

    # --------------------------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------------------------
    base_name = args.model_name.split("/")[-1]
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
    parser.add_argument("--config_path", type=str, default="./moe_model_metadata.json", help="Path to MoE model metadata JSON")
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
