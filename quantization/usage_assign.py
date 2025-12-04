import yaml
import torch
from typing import List, Dict


# -------------------------------------------------
# Config loader
# -------------------------------------------------
def load_usage_assign_config(path: str = "configs/bit_assign.yaml") -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# Unified entrypoint
# -------------------------------------------------
def assign_bits_from_usage(
    expert_counts: torch.Tensor,
    config_path: str = "configs/bit_assign.yaml",
    k_override: int = None,
    low_bits_override: int = None,
) -> List[List[int]]:
    """
    Unified dispatcher: chooses which quantization strategy
    to use based on config file.

    expert_counts: Tensor [num_layers, num_experts]
    k_override: If provided, overrides the k value in global_bottom_k mode
    low_bits_override: If provided, overrides the low_bits value
    return: bits[layer][expert]
    """
    cfg = load_usage_assign_config(config_path)
    mode = cfg.get("mode", "usage_bit_assign")

    if mode == "usage_bit_assign":
        # Override low_bits if provided
        percentile_cfg = cfg["usage_bit_assign"].copy()
        if low_bits_override is not None:
            print(f"   🔧 Overriding low_bits from config ({percentile_cfg.get('low_bits')}) with {low_bits_override}")
            percentile_cfg["low_bits"] = low_bits_override
        return _assign_bits_percentile(expert_counts, percentile_cfg)

    elif mode == "global_bottom_k":
        # Override k and/or low_bits if provided
        bottom_k_cfg = cfg["global_bottom_k"].copy()
        if k_override is not None:
            print(f"   🔧 Overriding k from config ({bottom_k_cfg.get('k')}) with {k_override}")
            bottom_k_cfg["k"] = k_override
        if low_bits_override is not None:
            print(f"   🔧 Overriding low_bits from config ({bottom_k_cfg.get('low_bits')}) with {low_bits_override}")
            bottom_k_cfg["low_bits"] = low_bits_override
        return _assign_bits_global_bottom_k(expert_counts, bottom_k_cfg)

    else:
        raise ValueError(f"Unknown bit assignment mode: {mode}")


# -------------------------------------------------
# Mode 1 — Per-layer percentile assignment
# -------------------------------------------------
def _assign_bits_percentile(
    expert_counts: torch.Tensor,
    cfg: Dict
) -> List[List[int]]:
    top_pct = cfg["top_percentile"]
    low_pct = cfg["low_percentile"]
    high_bits = cfg["high_bits"]
    mid_bits = cfg["mid_bits"]
    low_bits = cfg["low_bits"]

    num_layers, num_experts = expert_counts.shape
    bit_assignments = []

    for layer_idx in range(num_layers):
        usage = expert_counts[layer_idx].float()
        total = usage.sum()

        if total <= 0:
            # No routing in this layer → treat as non-MoE / full precision
            bit_assignments.append([high_bits] * num_experts)
            continue

        usage_norm = usage / total
        high_thr = torch.quantile(usage_norm, top_pct)
        low_thr = torch.quantile(usage_norm, low_pct)

        layer_bits = []
        for u in usage_norm:
            if u >= high_thr:
                layer_bits.append(high_bits)
            elif u >= low_thr:
                layer_bits.append(mid_bits)
            else:
                layer_bits.append(low_bits)

        bit_assignments.append(layer_bits)

    return bit_assignments


# -------------------------------------------------
# Mode 2 — Global bottom-K assignment
# -------------------------------------------------
def _assign_bits_global_bottom_k(
    expert_counts: torch.Tensor,
    cfg: Dict
) -> List[List[int]]:
    K = cfg["k"]
    low_bits = cfg["low_bits"]
    high_bits = cfg["high_bits"]

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

    # Select bottom-K
    quantized_set = set()
    selected = []

    for i in range(min(K, len(flat_usage))):
        usage, L, E = flat_usage[i]
        quantized_set.add((L, E))
        selected.append((usage, L, E))

    # 🔍 DEBUG PRINT
    print("\n🔍 Top 10 lowest-usage experts selected for quantization:")
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
