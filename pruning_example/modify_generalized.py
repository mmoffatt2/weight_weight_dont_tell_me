# more generic removal by sonnet 4.5
from transformers.dynamic_module_utils import resolve_trust_remote_code
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import gc
import os
import json
from typing import Dict, Set, Union, List, Tuple, Optional


class MoEModifierMemoryEfficient:
    def __init__(
        self,
        model_name: str,
        experts_to_remove: Optional[Union[int, List, Set]] = None,
        layer_specific_experts: Optional[Dict[int, Union[int, List, Set]]] = None,
    ):
        """
        Generic MoE model modifier with per-layer expert removal support

        Args:
            model_name: HuggingFace model name/path
            experts_to_remove: int, list, set - experts to remove from ALL layers
            layer_specific_experts: dict mapping layer_idx -> experts to remove
                IMPORTANT: When using layer_specific_experts, the final expert count
                must be the same across all layers for proper model reloading.

                Examples:
                    # All layers end up with 62 experts (64 - 2)
                    {0: [0, 1], 1: [3, 5], 2: [10, 15]}

                    # Invalid - different final counts (will raise error)
                    {0: [0, 1], 1: [3, 5, 7]}  # Layer 0 has 62, Layer 1 has 61
        """
        self.model_name = model_name
        self.config = None
        self.device = None
        self.expected_final_expert_count = None

        # Handle layer-specific or global expert removal
        if layer_specific_experts is not None:
            self.layer_specific_experts = self._normalize_layer_specific_experts(
                layer_specific_experts
            )
            self.mode = "layer_specific"
            print(f"🎯 Layer-specific mode: Will remove different experts per layer")
            for layer_idx, experts in sorted(self.layer_specific_experts.items()):
                print(f"   Layer {layer_idx}: Remove experts {sorted(experts)}")
        elif experts_to_remove is not None:
            self.experts_to_remove = self._normalize_experts_set(experts_to_remove)
            self.mode = "global"
            print(
                f"🎯 Global mode: Will remove {len(self.experts_to_remove)} "
                f"expert(s) from all layers: {sorted(self.experts_to_remove)}"
            )
        else:
            raise ValueError(
                "Must provide either experts_to_remove or layer_specific_experts"
            )

    def _normalize_experts_set(self, experts: Union[int, List, Tuple, Set]) -> Set[int]:
        """Convert int/list/tuple/set to set of expert indices"""
        if isinstance(experts, int):
            return {experts}
        elif isinstance(experts, (list, tuple)):
            return set(experts)
        elif isinstance(experts, set):
            return experts
        else:
            raise ValueError("experts must be int, list, tuple, or set")

    def _normalize_layer_specific_experts(
        self, layer_dict: Dict[int, Union[int, List, Tuple, Set]]
    ) -> Dict[int, Set[int]]:
        """Normalize layer_specific_experts dict to have sets as values"""
        normalized = {}
        for layer_idx, experts in layer_dict.items():
            if not isinstance(layer_idx, int):
                raise ValueError(f"Layer index must be int, got {type(layer_idx)}")
            normalized[layer_idx] = self._normalize_experts_set(experts)
        return normalized

    def get_experts_to_remove_for_layer(self, layer_idx: int) -> Set[int]:
        """Get the set of experts to remove for a specific layer"""
        if self.mode == "layer_specific":
            return self.layer_specific_experts.get(layer_idx, set())
        else:
            return self.experts_to_remove

    def validate_consistent_expert_count(
        self, original_num_experts: int, num_layers: int
    ):
        """
        Validate that all layers will end up with the same expert count.
        This is required for proper model reloading with HuggingFace.
        """
        if self.mode == "global":
            # Global mode always results in consistent counts
            self.expected_final_expert_count = original_num_experts - len(
                self.experts_to_remove
            )
            print(
                f"\n✅ Validation passed: All layers will have "
                f"{self.expected_final_expert_count} experts"
            )
            return

        # Layer-specific mode: check consistency
        print("\n🔍 Validating expert count consistency...")

        # Calculate expected counts for all layers
        expert_counts = {}
        removal_counts = {}

        for layer_idx in range(num_layers):
            experts_to_remove = self.get_experts_to_remove_for_layer(layer_idx)
            num_removed = len(experts_to_remove)
            final_count = original_num_experts - num_removed

            expert_counts[layer_idx] = final_count
            removal_counts[layer_idx] = num_removed

        # Check if all counts are the same
        unique_counts = set(expert_counts.values())

        if len(unique_counts) > 1:
            # Build detailed error message
            error_msg = [
                "\n❌ ERROR: Inconsistent expert counts detected!",
                f"   Original experts per layer: {original_num_experts}",
                "\n   Per-layer breakdown:",
            ]

            for layer_idx in range(num_layers):
                removed = removal_counts[layer_idx]
                final = expert_counts[layer_idx]
                if layer_idx in self.layer_specific_experts:
                    experts_list = sorted(self.layer_specific_experts[layer_idx])
                    error_msg.append(
                        f"     Layer {layer_idx}: Remove {removed} {experts_list} "
                        f"→ {final} experts"
                    )
                else:
                    error_msg.append(
                        f"     Layer {layer_idx}: Remove {removed} → {final} experts"
                    )

            error_msg.extend(
                [
                    f"\n   Final counts: {sorted(unique_counts)}",
                    "\n   SOLUTION: Ensure all layers remove the same NUMBER of experts.",
                    "   Example (all remove 2 experts):",
                    "     {0: [0, 1], 1: [3, 5], 2: [10, 15]}",
                ]
            )

            raise ValueError("\n".join(error_msg))

        # All counts are consistent
        self.expected_final_expert_count = expert_counts[0]
        print(
            f"✅ Validation passed: All {num_layers} layers will have "
            f"{self.expected_final_expert_count} experts"
        )

        # Show summary
        print("\n📊 Removal summary:")
        for layer_idx, removed in removal_counts.items():
            if removed > 0:
                experts_list = sorted(self.layer_specific_experts[layer_idx])
                print(f"   Layer {layer_idx}: Remove {removed} experts {experts_list}")

    def setup_device(self):
        """Setup device with fallbacks"""
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(f"✅ MPS device available: {self.device}")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"✅ CUDA device available: {self.device}")
        else:
            self.device = torch.device("cpu")
            print("⚠️ Using CPU")
        return self.device

    def remove_accelerate_hooks(self, module):
        """Recursively remove accelerate hooks from a module"""
        if hasattr(module, "_hf_hook"):
            delattr(module, "_hf_hook")
        for child in module.children():
            self.remove_accelerate_hooks(child)

    def detect_moe_structure(self, layer):
        """
        Detect MoE structure in a layer
        Returns: (mlp_module, experts_module, gate_module) or (None, None, None)
        """
        patterns = [
            ("mlp", "experts", "gate"),
            ("block_sparse_moe", "experts", "gate"),
            ("moe", "experts", "router"),
            ("feed_forward", "experts", "gate"),
        ]

        for mlp_attr, experts_attr, gate_attr in patterns:
            if hasattr(layer, mlp_attr):
                mlp_module = getattr(layer, mlp_attr)
                if hasattr(mlp_module, experts_attr):
                    experts_module = getattr(mlp_module, experts_attr)
                    gate_module = None
                    for gate_name in [gate_attr, "gate", "router"]:
                        if hasattr(mlp_module, gate_name):
                            gate_module = getattr(mlp_module, gate_name)
                            break
                    return mlp_module, experts_module, gate_module

        return None, None, None

    def modify_and_save_layer_by_layer(
        self, output_dir: str, torch_dtype=torch.float16
    ):
        """Memory-efficient: Load, modify, and save each layer individually"""
        print("\n🔧 Starting layer-by-layer modification...")

        # Load config
        self.config = AutoConfig.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Get original expert count
        if not hasattr(self.config, "num_experts"):
            raise ValueError(
                "Config does not have 'num_experts' attribute. "
                "This model may not be a MoE model or uses a different structure."
            )

        original_num_experts = self.config.num_experts
        print(f"📊 Original config: {original_num_experts} experts per layer")

        # Load model first to get number of layers
        print("\nLoading model with AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Detect model structure
        print("🔍 Detecting model structure...")
        base_model = model.model if hasattr(model, "model") else model

        if not hasattr(base_model, "layers"):
            raise ValueError(
                "Could not find 'layers' attribute in model. "
                "This model may not be supported."
            )

        num_layers = len(base_model.layers)
        print(f"📊 Model has {num_layers} layers")

        # CRITICAL: Validate that all layers will have the same final expert count
        self.validate_consistent_expert_count(original_num_experts, num_layers)

        # Update config with the final expert count
        self.config.num_experts = self.expected_final_expert_count
        print(
            f"\n📊 Config will be updated: {original_num_experts} → "
            f"{self.config.num_experts} experts"
        )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        tokenizer.save_pretrained(output_dir)

        # Process each layer
        print(f"\n{'=' * 60}")
        print(f"Processing {num_layers} layers...")
        print(f"{'=' * 60}")
        modified_layers = 0
        layer_expert_counts = {}

        for layer_idx, layer in enumerate(base_model.layers):
            print(f"\n📦 Layer {layer_idx}/{num_layers - 1}")

            experts_to_remove_this_layer = self.get_experts_to_remove_for_layer(
                layer_idx
            )

            if not experts_to_remove_this_layer:
                print(f"  ⏭️  No experts to remove from this layer")
                # Still track the count for verification
                mlp_module, experts_module, gate_module = self.detect_moe_structure(
                    layer
                )
                if experts_module is not None:
                    layer_expert_counts[layer_idx] = len(experts_module)
                continue

            layer = layer.to(self.device)

            mlp_module, experts_module, gate_module = self.detect_moe_structure(layer)

            if experts_module is not None:
                num_experts = len(experts_module)
                print(f"  ✓ Found MoE layer with {num_experts} experts")

                if max(experts_to_remove_this_layer) >= num_experts:
                    raise ValueError(
                        f"Layer {layer_idx}: Expert index "
                        f"{max(experts_to_remove_this_layer)} out of range "
                        f"(layer has {num_experts} experts)"
                    )

                # Remove experts
                new_experts = nn.ModuleList()
                for i in range(num_experts):
                    if i not in experts_to_remove_this_layer:
                        new_experts.append(experts_module[i])

                if hasattr(mlp_module, "experts"):
                    mlp_module.experts = new_experts

                print(
                    f"  ✓ Removed {len(experts_to_remove_this_layer)} expert(s): "
                    f"{sorted(experts_to_remove_this_layer)}"
                )
                print(f"  ✓ New expert count: {len(new_experts)}")

                # Verify count matches expectation
                if len(new_experts) != self.expected_final_expert_count:
                    raise RuntimeError(
                        f"Layer {layer_idx}: Expected {self.expected_final_expert_count} "
                        f"experts but got {len(new_experts)}. This should not happen!"
                    )

                layer_expert_counts[layer_idx] = len(new_experts)

                if gate_module is not None and hasattr(gate_module, "weight"):
                    self._adjust_router(
                        gate_module,
                        num_experts,
                        len(new_experts),
                        experts_to_remove_this_layer,
                    )

                modified_layers += 1

            layer = layer.to("cpu")

            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        print(f"\n{'=' * 60}")
        print(f"✅ Modified {modified_layers} MoE layers")
        print(f"{'=' * 60}")

        # Final verification
        print("\n🔍 Final verification:")
        unique_counts = set(layer_expert_counts.values())
        if len(unique_counts) != 1:
            raise RuntimeError(
                f"CRITICAL: Final expert counts are inconsistent: {unique_counts}. "
                "This should not happen after validation!"
            )
        print(
            f"✅ All {len(layer_expert_counts)} MoE layers have "
            f"{self.expected_final_expert_count} experts"
        )

        print("\n🪝 Removing accelerate hooks...")
        self.remove_accelerate_hooks(model)

        print("\n💾 Saving modified model...")
        model.save_pretrained(output_dir)
        self.config.save_pretrained(output_dir)

        # Save detailed expert modification info
        expert_info = {
            "mode": self.mode,
            "original_num_experts": original_num_experts,
            "final_num_experts": self.expected_final_expert_count,
            "num_layers": num_layers,
            "layer_expert_counts": layer_expert_counts,
        }

        if self.mode == "layer_specific":
            expert_info["removed_experts_per_layer"] = {
                k: sorted(list(v)) for k, v in self.layer_specific_experts.items()
            }
        else:
            expert_info["removed_experts_global"] = sorted(list(self.experts_to_remove))

        info_path = os.path.join(output_dir, "expert_modification_info.json")
        with open(info_path, "w") as f:
            json.dump(expert_info, f, indent=2)
        print(f"💾 Saved expert modification info to {info_path}")

        print("\n✅ Layer-by-layer modification complete!")
        print(
            f"✅ Model can be reloaded normally with AutoModelForCausalLM.from_pretrained()"
        )
        return model

    def _adjust_router(
        self,
        router,
        old_num_experts: int,
        new_num_experts: int,
        experts_to_remove: Set[int],
    ):
        """Adjust router layer dimensions to account for removed experts"""
        print(f"  Adjusting router: {old_num_experts} → {new_num_experts} outputs")

        original_weight = router.weight.data
        original_bias = router.bias.data if router.bias is not None else None

        new_weight = torch.zeros(
            new_num_experts,
            router.in_features,
            dtype=original_weight.dtype,
            device=original_weight.device,
        )

        new_idx = 0
        for i in range(old_num_experts):
            if i not in experts_to_remove:
                new_weight[new_idx] = original_weight[i]
                new_idx += 1

        router.out_features = new_num_experts
        router.weight = nn.Parameter(new_weight)

        if original_bias is not None:
            new_bias = torch.zeros(
                new_num_experts,
                dtype=original_bias.dtype,
                device=original_bias.device,
            )
            new_idx = 0
            for i in range(old_num_experts):
                if i not in experts_to_remove:
                    new_bias[new_idx] = original_bias[i]
                    new_idx += 1
            router.bias = nn.Parameter(new_bias)

        print(f"  ✓ Router adjusted successfully")


def main():
    """Example usage with different modes"""

    MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
    OUTPUT_DIR = "./moe_modified_efficient"

    print("=" * 60)
    print("MoE Expert Removal - Memory Efficient Mode")
    print("=" * 60)

    # ========================================
    # EXAMPLE 1: Global removal (same experts from all layers)
    # ========================================
    # print("\n" + "=" * 60)
    # print("Example 1: Global Expert Removal")
    # print("=" * 60)

    # modifier = MoEModifierMemoryEfficient(
    #     model_name=MODEL_NAME,
    #     experts_to_remove=[0, 1],  # Remove experts 0, 1 from ALL layers
    # )

    # ========================================
    # EXAMPLE 2: Layer-specific with SAME final count
    # ========================================
    # Uncomment to use this mode:
    print("\n" + "=" * 60)
    print("Example 2: Layer-Specific Expert Removal (Consistent Counts)")
    print("=" * 60)

    # All layers remove 2 experts (different ones, but same count!)
    modifier = MoEModifierMemoryEfficient(
        model_name=MODEL_NAME,
        layer_specific_experts={
            0: [0, 1],  # Layer 0: Remove experts 0, 1
            1: [3, 5],  # Layer 1: Remove experts 3, 5
            2: [10, 15],  # Layer 2: Remove experts 10, 15
            3: [2, 7],  # Layer 3: Remove experts 2, 7
            4: [0, 1],  # Layer 4: Remove experts 0, 1
            5: [3, 5],  # Layer 5: Remove experts 3, 5
            6: [10, 15],  # Layer 6: Remove experts 10, 15
            7: [2, 7],  # Layer 7: Remove experts 2, 7
            8: [0, 1],  # Layer 8: Remove experts 0, 1
            9: [3, 5],  # Layer 9: Remove experts 3, 5
            10: [10, 15],  # Layer 10: Remove experts 10, 15
            11: [2, 7],  # Layer 11: Remove experts 2, 7
            12: [0, 1],  # Layer 12: Remove experts 0, 1
            13: [3, 5],  # Layer 13: Remove experts 3, 5
            14: [10, 15],  # Layer 14: Remove experts 10, 15
            15: [
                2,
                7,
            ],  # Layer 15: Remove experts 2, 7 Other layers: no removal (will keep all experts)
            # If you want to remove from all layers, specify all layer indices
        },
    )

    # ========================================
    # EXAMPLE 3: This will FAIL (inconsistent counts)
    # ========================================
    # Uncomment to see validation error:
    """
    modifier = MoEModifierMemoryEfficient(
        model_name=MODEL_NAME,
        layer_specific_experts={
            0: [0, 1],      # Removes 2 experts
            1: [3, 5, 7],   # Removes 3 experts - INCONSISTENT!
        }
    )
    # This will raise ValueError during validation
    """

    device = modifier.setup_device()

    model = modifier.modify_and_save_layer_by_layer(
        output_dir=OUTPUT_DIR, torch_dtype=torch.float16
    )

    # Test
    print("\n" + "=" * 60)
    print("🧪 Testing modified model...")
    print("=" * 60)
    try:
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
        config = AutoConfig.from_pretrained(OUTPUT_DIR, trust_remote_code=True)

        print(f"✓ Config loaded: num_experts = {config.num_experts}")

        test_model = AutoModelForCausalLM.from_pretrained(
            OUTPUT_DIR,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        print("✓ Model loaded successfully!")

        test_prompt = "The future of AI is"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(test_model.device)

        with torch.no_grad():
            outputs = test_model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        print(f"\nPrompt: {test_prompt}")
        print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        print("\n✅ Test successful! Model works perfectly with HuggingFace!")

    except Exception as e:
        print(f"⚠️ Test failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n🧹 Cleaning up...")
    del model
    if "test_model" in locals():
        del test_model
    gc.collect()

    print("\n🎉 Done! Modified model saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
