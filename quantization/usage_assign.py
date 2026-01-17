import random
import torch
from typing import List

def assign_bits_from_usage(
    expert_counts: torch.Tensor,
    K: int,
    low_bits: int,
    high_bits: int,
    strategy: str = "bottom_k",
) -> List[List[int]]:
    
    if K is None or K < 0:
        raise ValueError(f"K must be a non-negative int, got {K}")

    num_layers, num_experts = expert_counts.shape

    # Flatten (usage, layer, expert) but SKIP layers with no routing
    flat_usage = []
    for layer_idx in range(num_layers):
        layer_total = expert_counts[layer_idx].sum().item()
        if layer_total == 0:
            continue
        for expert_idx in range(num_experts):
            usage = expert_counts[layer_idx, expert_idx].item()
            flat_usage.append((usage, layer_idx, expert_idx))

    if len(flat_usage) == 0:
        print("⚠️ No routed experts found — defaulting all to high_bits")
        return [[high_bits] * num_experts for _ in range(num_layers)]

    # Sort by usage ascending
    flat_usage.sort(key=lambda x: x[0])

    # Select K experts based on strategy
    quantized_set = set()
    selected = []

    num_selected = min(K, len(flat_usage))

    if strategy == "bottom_k":
        chosen = flat_usage[:num_selected]

    elif strategy == "top_k":
        chosen = flat_usage[-num_selected:]          # highest-usage K
        # optional: sort in descending order for nicer logging
        chosen = sorted(chosen, key=lambda x: x[0], reverse=True)

    elif strategy == "random_k":
        # random distinct (usage, layer, expert) tuples
        chosen = random.sample(flat_usage, k=num_selected)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    for usage, L, E in chosen:
        quantized_set.add((L, E))
        selected.append((usage, L, E))


    # 🔍 DEBUG PRINT
    print("\n🔍 First 10 experts selected for quantization with strategy:", strategy, "\n")
    for usage, L, E in selected[:10]:
        print(f"  Layer {L:02d} | Expert {E:02d} | usage={int(usage)}")

    # Build output
    bit_assignments: List[List[int]] = []
    for layer_idx in range(num_layers):
        layer_total = expert_counts[layer_idx].sum().item()

        if layer_total == 0:
            bit_assignments.append([high_bits] * num_experts)
            continue

        layer_bits = []
        for expert_idx in range(num_experts):
            if (layer_idx, expert_idx) in quantized_set:
                layer_bits.append(low_bits)
            else:
                layer_bits.append(high_bits)
        bit_assignments.append(layer_bits)

    return bit_assignments
