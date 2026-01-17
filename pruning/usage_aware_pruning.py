# pruning/usage_aware_pruning.py

import sys
import os
import argparse
import json
import random
import torch
from typing import Dict, Set, Optional, Union, List

# ---- Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def load_expert_counts_and_select_bottom_k(
    expert_counts_path: str, top_k: int
) -> Dict[int, Set[int]]:
    """
    Load expert counts from a .pt file and select bottom-k experts per layer.

    Args:
        expert_counts_path: Path to expert_counts.pt file (tensor of shape [num_layers, num_experts])
        top_k: Number of least-used experts to remove per layer

    Returns:
        Dict mapping layer_idx -> set of expert indices to remove
    """
    print(f"\n📊 Loading expert counts from: {expert_counts_path}")
    expert_counts = torch.load(expert_counts_path, map_location="cpu")

    num_layers, num_experts = expert_counts.shape
    print(f"   Shape: [{num_layers} layers × {num_experts} experts]")
    print(f"   Will remove bottom {top_k} experts from each layer")

    layer_specific_experts = {}

    for layer_idx in range(num_layers):
        layer_counts = expert_counts[layer_idx]

        # Skip layers with no routing (e.g., dense layers)
        if layer_counts.sum() == 0:
            print(f"   Layer {layer_idx}: No routing detected, skipping")
            continue

        # Sort experts by usage (ascending)
        sorted_indices = torch.argsort(layer_counts, descending=False)

        # Select bottom-k experts to remove
        experts_to_remove = sorted_indices[:top_k].tolist()
        layer_specific_experts[layer_idx] = set(experts_to_remove)

        # Show which experts will be removed
        removed_usages = [layer_counts[i].item() for i in experts_to_remove]
        print(
            f"   Layer {layer_idx}: Remove experts {experts_to_remove} (usage: {removed_usages})"
        )

    return layer_specific_experts


def load_expert_counts_and_select_top_k(
    expert_counts_path: str, top_k: int
) -> Dict[int, Set[int]]:
    """
    Load expert counts from a .pt file and select top-k experts per layer.

    Args:
        expert_counts_path: Path to expert_counts.pt file (tensor of shape [num_layers, num_experts])
        top_k: Number of most-used experts to remove per layer

    Returns:
        Dict mapping layer_idx -> set of expert indices to remove
    """
    print(f"\n📊 Loading expert counts from: {expert_counts_path}")
    expert_counts = torch.load(expert_counts_path, map_location="cpu")

    num_layers, num_experts = expert_counts.shape
    print(f"   Shape: [{num_layers} layers × {num_experts} experts]")
    print(f"   Will remove top {top_k} experts from each layer")

    layer_specific_experts = {}

    for layer_idx in range(num_layers):
        layer_counts = expert_counts[layer_idx]

        # Skip layers with no routing (e.g., dense layers)
        if layer_counts.sum() == 0:
            print(f"   Layer {layer_idx}: No routing detected, skipping")
            continue

        # Sort experts by usage (descending)
        sorted_indices = torch.argsort(layer_counts, descending=True)

        # Select top-k experts to remove
        experts_to_remove = sorted_indices[:top_k].tolist()
        layer_specific_experts[layer_idx] = set(experts_to_remove)

        # Show which experts will be removed
        removed_usages = [layer_counts[i].item() for i in experts_to_remove]
        print(
            f"   Layer {layer_idx}: Remove experts {experts_to_remove} (usage: {removed_usages})"
        )

    return layer_specific_experts


def generate_random_expert_removal(
    num_layers: int, num_experts: int, k: int, seed: int = 42
) -> Dict[int, Set[int]]:
    """
    Generate random expert removal for each layer.

    Args:
        num_layers: Number of layers in the model
        num_experts: Number of experts per layer
        k: Number of experts to remove per layer
        seed: Random seed for reproducibility

    Returns:
        Dict mapping layer_idx -> set of expert indices to remove
    """
    print(f"\n🎲 Generating random expert removal:")
    print(f"   Layers: {num_layers}, Experts per layer: {num_experts}")
    print(f"   Will remove {k} random experts from each layer")
    print(f"   Using seed: {seed}")

    random.seed(seed)
    layer_specific_experts = {}

    for layer_idx in range(num_layers):
        # Generate k unique random expert indices
        experts_to_remove = random.sample(range(num_experts), k)
        layer_specific_experts[layer_idx] = set(experts_to_remove)
        print(
            f"   Layer {layer_idx}: Remove random experts {sorted(experts_to_remove)}"
        )

    return layer_specific_experts


def main():
    parser = argparse.ArgumentParser("MoE Usage-Aware Pruning")

    # Model and files
    parser.add_argument(
        "--model_name", type=str, required=True, help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the pruned model",
    )

    # Pruning configuration
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["bottom_k", "top_k", "random"],
        required=True,
        help="Pruning strategy",
    )
    parser.add_argument(
        "--k", type=int, required=True, help="Number of experts to remove per layer"
    )

    # Expert counts (for bottom_k and top_k strategies)
    parser.add_argument(
        "--expert_counts", type=str, help="Path to expert_counts.pt file from tracing"
    )

    # Dataset (for auto-generating counts if needed)
    parser.add_argument(
        "--dataset", type=str, help="Dataset name for auto-generating expert counts"
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (for random strategy)",
    )

    args = parser.parse_args()

    # Import after argument parsing
    from modify_generalized import (
        MoEModifierMemoryEfficient,
        generate_expert_counts_via_tracing,
    )

    print("=" * 60)
    print("MoE Usage-Aware Pruning")
    print("=" * 60)
    print(f"Model:    {args.model_name}")
    print(f"Strategy: {args.strategy}")
    print(f"K:        {args.k} experts per layer")
    print(f"Output:   {args.output_dir}")
    print("=" * 60)

    # Determine expert removal based on strategy
    if args.strategy == "bottom_k":
        if not args.expert_counts and not args.dataset:
            print("❌ Error: bottom_k strategy requires --expert_counts or --dataset")
            return 1

        if not args.expert_counts:
            # Auto-generate counts via tracing
            print("🔄 Auto-generating expert counts via tracing...")
            args.expert_counts = generate_expert_counts_via_tracing(
                args.model_name, args.dataset, args.output_dir
            )

        layer_specific_experts = load_expert_counts_and_select_bottom_k(
            args.expert_counts, args.k
        )

    elif args.strategy == "top_k":
        if not args.expert_counts and not args.dataset:
            print("❌ Error: top_k strategy requires --expert_counts or --dataset")
            return 1

        if not args.expert_counts:
            # Auto-generate counts via tracing
            print("🔄 Auto-generating expert counts via tracing...")
            args.expert_counts = generate_expert_counts_via_tracing(
                args.model_name, args.dataset, args.output_dir
            )

        layer_specific_experts = load_expert_counts_and_select_top_k(
            args.expert_counts, args.k
        )

    elif args.strategy == "random":
        # Need to get model info to determine num_layers and num_experts
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

        num_layers = (
            config.num_hidden_layers
            if hasattr(config, "num_hidden_layers")
            else getattr(config, "n_layers", 28)
        )
        num_experts = (
            config.num_experts
            if hasattr(config, "num_experts")
            else getattr(config, "n_routed_experts", 64)
        )

        layer_specific_experts = generate_random_expert_removal(
            num_layers, num_experts, args.k, args.seed
        )

    else:
        print(f"❌ Error: Unknown strategy {args.strategy}")
        return 1

    # Create modifier and apply pruning
    print("\n" + "=" * 60)
    print("Applying Pruning")
    print("=" * 60)

    modifier = MoEModifierMemoryEfficient(
        model_name=args.model_name,
        layer_specific_experts=layer_specific_experts,
    )

    # Setup device and apply modifications
    device = modifier.setup_device()
    model = modifier.modify_and_save_layer_by_layer(
        output_dir=args.output_dir, torch_dtype=torch.float16
    )

    # Save pruning metadata
    pruning_info = {
        "strategy": args.strategy,
        "k": args.k,
        "model_name": args.model_name,
        "seed": args.seed if args.strategy == "random" else None,
        "expert_counts": args.expert_counts
        if (args.strategy in ["bottom_k", "top_k"])
        else None,
        "dataset": args.dataset,
    }

    pruning_info_path = os.path.join(args.output_dir, "pruning_info.json")
    with open(pruning_info_path, "w") as f:
        json.dump(pruning_info, f, indent=2)
    print(f"💾 Saved pruning info to {pruning_info_path}")

    print("\n✅ Pruning complete!")
    print(f"✅ Pruned model saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
