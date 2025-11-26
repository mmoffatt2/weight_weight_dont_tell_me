# quantization/build_bitdict.py
from typing import Dict, List


def build_deepseek_bitdict(
    bit_assignments: List[List[int]],
    num_layers: int,
    num_experts: int,
) -> Dict[str, int]:
    """
    Build bit dict for DeepSeek MoE:

    Only MoE expert MLPs are listed in the dict.
    All other Linear layers are implicitly skipped.
    """

    bitdict: Dict[str, int] = {}

    for layer_idx in range(num_layers):

        # Skip layer 0 (dense MLP, no experts)
        if layer_idx == 0:
            continue

        # MoE Experts only
        for expert_idx in range(num_experts):
            bit = bit_assignments[layer_idx][expert_idx]
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                bitdict[
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}"
                ] = bit

    return bitdict
