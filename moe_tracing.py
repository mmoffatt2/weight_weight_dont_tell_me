# -*- coding: utf-8 -*-
"""
Universal MoE Router Tracing Utility (Metadata Only)
----------------------------------------------------
Author: Michael Moffatt
Date: 2025-10-12

Usage:
    python moe_router_trace.py \
        --model_name mistralai/Mixtral-8x7B-v0.1 \
        --config_path ./configs/moe_model_metadata.json \
        --save_dir ./results \
        --nsamples 32 \
        --seqlen 2048 \
        --batch_size 1
"""

import os
import random
import json
import argparse
import torch
from datasets import load_dataset, Dataset
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
# Dataset helper
# ------------------------------------------------------------------------------
def get_wikitext2(tokenizer, seqlen: int, nsamples: int, split: str = "train"):
    """Randomly slice WikiText-2 sequences."""
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train" if split == "train" else "test")
    text = "".join([" \n" if s == "" else s for s in data["text"][:1000]])
    enc = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attn = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attn})
    return dataset

# ------------------------------------------------------------------------------
# Main tracing function
# ------------------------------------------------------------------------------
def dump_routing_trace(args):
    """Run an MoE model on WikiText-2 and record routing decisions."""
    os.makedirs(args.save_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # Load model metadata
    # --------------------------------------------------------------------------
    with open(args.config_path) as f:
        metadata = json.load(f)

    if args.model_name not in metadata:
        raise ValueError(f"Model {args.model_name} not found in {args.config_path}")

    model_info = metadata[args.model_name]
    router_key = model_info["router_key"]
    num_layers = model_info["num_hidden_layers"]
    num_experts = model_info["num_routed_experts"]
    outputs_logits = model_info["outputs_logits"]
    top_k = model_info["num_experts_per_token"]

    print(f"✅ Using metadata for {args.model_name}")
    print(f"   Layers={num_layers}, Experts={num_experts}, Top-k={top_k}, Router key='{router_key}'")

    # --------------------------------------------------------------------------
    # Load tokenizer, data, and model
    # --------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = get_wikitext2(tokenizer, seqlen=args.seqlen, nsamples=args.nsamples)
    data_loader = DataLoader(
        Dataset.from_list(dataset),
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        shuffle=True,
    )

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")

    expert_counts = torch.zeros(num_layers, num_experts)
    per_token_records = []

    print(f"Tracing routing decisions...")

    # --------------------------------------------------------------------------
    # Inference loop
    # --------------------------------------------------------------------------
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Dumping routing trace")):
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, output_router_logits=True)

        # Extract router data using the known key
        if not hasattr(outputs, router_key):
            raise KeyError(f"Router key '{router_key}' not found in model outputs.")

        router_data = getattr(outputs, router_key)
        if isinstance(router_data, (list, tuple)):
            router_data = torch.stack(router_data)

        probs = torch.softmax(router_data, dim=-1) if outputs_logits else router_data

        # Count expert usage and record per-token info
        topk_values, topk_indices = torch.topk(probs, k=top_k, dim=-1)
        for layer_idx in range(num_layers):
            unique, counts = torch.unique(topk_indices[layer_idx], return_counts=True)
            expert_counts[layer_idx, unique.cpu()] += counts.cpu()

        per_token_records.append({
            "batch_idx": batch_idx,
            "topk_indices": topk_indices.cpu().tolist(),
            "topk_values": topk_values.cpu().tolist()
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
    parser = argparse.ArgumentParser(description="Universal MoE Router Tracing Utility (metadata only)")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model ID (must exist in metadata JSON)")
    parser.add_argument("--config_path", type=str, required=True, help="Path to MoE model metadata JSON")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save routing data")
    parser.add_argument("--nsamples", type=int, default=64, help="Number of random sequences to test")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length per sample")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")

    args = parser.parse_args()
    dump_routing_trace(args)
