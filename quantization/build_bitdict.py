# quantization/build_bitdict.py
from typing import Dict, List


def build_deepseek_bitdict(
    bit_assignments: List[List[int]],
    num_layers: int,
    num_experts: int,
    quantize_threshold: int = 8,
) -> Dict[str, int]:
    """
    Build bit dict for DeepSeek MoE:

    Only MoE expert MLPs are listed in the dict.
    Only tensors with bits <= quantize_threshold are included.
    Tensors with bits > quantize_threshold are left in full precision.
    
    This ensures we only pass valid GPTQ quantization targets (2,3,4,8 bits).
    """

    bitdict: Dict[str, int] = {}
    skipped_count = 0

    for layer_idx in range(num_layers):

        # Skip layer 0 (dense MLP, no experts)
        if layer_idx == 0:
            continue

        # MoE Experts only
        for expert_idx in range(num_experts):
            bit = bit_assignments[layer_idx][expert_idx]
            
            # Only include if quantization is needed (bit <= threshold)
            if bit <= quantize_threshold:
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    bitdict[
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}"
                    ] = bit
            else:
                skipped_count += 3  # 3 projections per expert

    if skipped_count > 0:
        print(f"ℹ️  Skipped {skipped_count} tensors above {quantize_threshold}-bit threshold (kept at full precision)")

    return bitdict


def build_mixtral_bitdict(
    bit_assignments: List[List[int]],
    num_layers: int,
    num_experts: int,
    quantize_threshold: int = 8,
) -> Dict[str, int]:
    """
    Build bit dict for Mixtral MoE:

    - Targets expert MLP projections under `block_sparse_moe.experts`.
    - Mixtral experts commonly expose three projection tensors: `w1`, `w2`, `w3`.
    - Only tensors with bits <= quantize_threshold are included; others remain full-precision.

    This produces a mapping suitable for GPTQ quantization (2,3,4,8 bits).
    """

    bitdict: Dict[str, int] = {}
    skipped_count = 0

    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            bit = bit_assignments[layer_idx][expert_idx]

            if bit <= quantize_threshold:
                for proj in ["w1", "w2", "w3"]:
                    bitdict[
                        f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.{proj}"
                    ] = bit
            else:
                skipped_count += 3  # 3 projections per expert

    if skipped_count > 0:
        print(
            f"ℹ️  Skipped {skipped_count} tensors above {quantize_threshold}-bit threshold (kept at full precision)"
        )

    return bitdict
