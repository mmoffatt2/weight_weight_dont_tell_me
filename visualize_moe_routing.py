# -*- coding: utf-8 -*-
"""
MoE Routing Visualization Utility
---------------------------------
Author: Michael Moffatt
Date: 2025-10-13

Usage:
    python visualize_moe_routing.py --results_dir ./results --model_name mistralai/Mixtral-8x7B-v0.1
"""

import os
import json
import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

sns.set_context("talk")
sns.set_style("whitegrid")

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def load_expert_counts(results_dir, model_name):
    """Load expert_counts.pt"""
    base_name = model_name.split("/")[-1]
    path = os.path.join(results_dir, f"{base_name}_expert_counts.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    expert_counts = torch.load(path, map_location="cpu")
    return expert_counts


def load_token_trace(results_dir, model_name):
    """Load token_trace.json"""
    base_name = model_name.split("/")[-1]
    path = os.path.join(results_dir, f"{base_name}_token_trace.json")
    if not os.path.exists(path):
        print(f"⚠️  No token_trace.json found at {path}. Skipping token-level analysis.")
        return None
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------------------
# Visualization utilities
# ------------------------------------------------------------------------------

def plot_expert_heatmap(expert_counts, model_name, save_dir):
    """Plot normalized expert usage per layer."""
    counts_norm = expert_counts / (expert_counts.sum(dim=1, keepdim=True) + 1e-8)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        counts_norm.numpy(),
        cmap="viridis",
        xticklabels=[f"E{i}" for i in range(counts_norm.shape[1])],
        yticklabels=[f"L{i}" for i in range(counts_norm.shape[0])],
        cbar_kws={"label": "Normalized routing frequency"},
    )
    plt.title(f"Expert Usage per Layer ({model_name})")
    plt.xlabel("Expert ID")
    plt.ylabel("Layer")
    plt.tight_layout()
    out_path = os.path.join(save_dir, "expert_heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved heatmap: {out_path}")


def plot_expert_bar(expert_counts, model_name, save_dir):
    """Plot total expert usage across all layers."""
    total_usage = expert_counts.sum(dim=0)
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(total_usage)), total_usage.numpy())
    plt.xlabel("Expert ID")
    plt.ylabel("Total tokens routed")
    plt.title(f"Global Expert Load ({model_name})")
    plt.tight_layout()
    out_path = os.path.join(save_dir, "expert_bar.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved global expert bar chart: {out_path}")


def plot_entropy_trace(token_trace, model_name, save_dir):
    """Plot average routing entropy per layer (if topk_values exist)."""
    if not token_trace or "topk_values" not in token_trace[0]:
        print("⚠️  No topk_values found — skipping entropy plot.")
        return

    entropy_per_layer = defaultdict(list)

    for record in token_trace:
        probs = np.array(record["topk_values"])
        # Flatten across tokens and top-k
        p = probs.reshape(-1, probs.shape[-1])
        entropy = -np.sum(p * np.log(p + 1e-8), axis=-1).mean()
        entropy_per_layer[record["layer_name"]].append(entropy)

    # Average across batches
    layers = list(entropy_per_layer.keys())
    entropies = [np.mean(entropy_per_layer[l]) for l in layers]

    plt.figure(figsize=(10, 4))
    plt.plot(layers, entropies, marker="o")
    plt.xticks(rotation=90)
    plt.ylabel("Average Routing Entropy")
    plt.title(f"Routing Sharpness per Layer ({model_name})")
    plt.tight_layout()
    out_path = os.path.join(save_dir, "routing_entropy.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved entropy plot: {out_path}")

def plot_expert_usage_count(expert_counts, model_name, save_dir):
    """Visualize absolute expert usage frequency (no normalization or log scale)."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        expert_counts.numpy(),
        cmap="YlGnBu",
        xticklabels=[f"E{i}" for i in range(expert_counts.shape[1])],
        yticklabels=[f"L{i}" for i in range(expert_counts.shape[0])],
        cbar_kws={"label": "Number of tokens routed"},
    )
    plt.title(f"Absolute Expert Usage per Layer — {model_name}")
    plt.xlabel("Expert ID")
    plt.ylabel("Layer")
    plt.tight_layout()
    out_path = os.path.join(save_dir, "expert_usage_counts.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved expert usage count map: {out_path}")

def plot_expert_sparsity(expert_counts, model_name, save_dir):
    counts_norm = expert_counts / (expert_counts.sum(dim=1, keepdim=True) + 1e-8)
    counts_log = torch.log10(counts_norm + 1e-6)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        counts_log.numpy(),
        cmap="magma_r",
        xticklabels=[f"E{i}" for i in range(counts_log.shape[1])],
        yticklabels=[f"L{i}" for i in range(counts_log.shape[0])],
        cbar_kws={"label": "log10(normalized routing freq)"}
    )
    plt.title(f"Expert Sparsity Map (Low = Underused) — {model_name}")
    plt.xlabel("Expert ID")
    plt.ylabel("Layer")
    plt.tight_layout()
    out_path = os.path.join(save_dir, "expert_sparsity_map.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved sparsity map: {out_path}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main(args):
    os.makedirs(args.results_dir, exist_ok=True)

    expert_counts = load_expert_counts(args.results_dir, args.model_name)
    token_trace = load_token_trace(args.results_dir, args.model_name)

    plot_expert_heatmap(expert_counts, args.model_name, args.results_dir)
    plot_expert_bar(expert_counts, args.model_name, args.results_dir)
    plot_expert_usage_count(expert_counts, args.model_name, args.results_dir)
    plot_expert_sparsity(expert_counts, args.model_name, args.results_dir)

    if token_trace:
        plot_entropy_trace(token_trace, args.model_name, args.results_dir)

    print(f"\n🎨 Visualization complete! All plots saved to {args.results_dir}")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MoE routing results")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory with tracing outputs")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g. mistralai/Mixtral-8x7B-v0.1)")
    args = parser.parse_args()
    main(args)
