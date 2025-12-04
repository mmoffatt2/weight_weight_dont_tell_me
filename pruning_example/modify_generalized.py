# more generic removal by sonnet 4.5
import sys
import os

# ---- Add MoE-Quantization to path BEFORE other imports for lm_eval
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "MoE-Quantization"))

from transformers.dynamic_module_utils import resolve_trust_remote_code
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import gc
import json
import argparse
import subprocess
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

        # Check if all counts are the same (only for layers that have removals)
        layers_with_removals = {k: v for k, v in expert_counts.items() if removal_counts[k] > 0}
        if not layers_with_removals:
            print("⚠️ No layers have expert removals specified")
            self.expected_final_expert_count = original_num_experts
            return
        
        unique_counts = set(layers_with_removals.values())

        if len(unique_counts) > 1:
            # Build detailed error message
            error_msg = [
                "\n❌ ERROR: Inconsistent expert counts detected!",
                f"   Original experts per layer: {original_num_experts}",
                "\n   Per-layer breakdown (layers with removals only):",
            ]

            for layer_idx in layers_with_removals.keys():
                removed = removal_counts[layer_idx]
                final = expert_counts[layer_idx]
                if layer_idx in self.layer_specific_experts:
                    experts_list = sorted(self.layer_specific_experts[layer_idx])
                    error_msg.append(
                        f"     Layer {layer_idx}: Remove {removed} {experts_list} "
                        f"→ {final} experts"
                    )

            error_msg.extend(
                [
                    f"\n   Final counts: {sorted(unique_counts)}",
                    "\n   SOLUTION: Ensure all layers with removals remove the same NUMBER of experts.",
                    "   Example (all remove 2 experts):",
                    "     {1: [0, 1], 2: [3, 5], 3: [10, 15]}",
                ]
            )

            raise ValueError("\n".join(error_msg))

        # All counts are consistent among layers with removals
        self.expected_final_expert_count = list(layers_with_removals.values())[0]
        num_layers_with_removals = len(layers_with_removals)
        print("layers_with_removals: ", layers_with_removals)
        print(
            f"✅ Validation passed: {num_layers_with_removals} layers with removals will have "
            f"{self.expected_final_expert_count} experts"
        )
        
        # Note about layers without removals
        num_layers_without_removals = num_layers - num_layers_with_removals
        if num_layers_without_removals > 0:
            print(f"   ({num_layers_without_removals} layers will keep all {original_num_experts} experts)")

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

        self.config.num_experts = self.config.n_routed_experts

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

        print("base_model: ", base_model)

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
        self.config.n_routed_experts = self.expected_final_expert_count
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

        # Final verification - check that layers with modifications match expected count
        print("\n🔍 Final verification:")
        modified_layer_counts = {k: v for k, v in layer_expert_counts.items() 
                                 if k in self.layer_specific_experts and len(self.layer_specific_experts[k]) > 0}
        if modified_layer_counts:
            unique_counts = set(modified_layer_counts.values())
            if len(unique_counts) != 1:
                raise RuntimeError(
                    f"CRITICAL: Final expert counts are inconsistent among modified layers: {unique_counts}. "
                    "This should not happen after validation!"
                )
            print(
                f"✅ All {len(modified_layer_counts)} modified MoE layers have "
                f"{self.expected_final_expert_count} experts"
            )
        else:
            print("⚠️ No layers were modified")

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
        """Adjust router layer dimensions to account for removed experts.

        Be robust to router implementations that may not expose bias/out_features
        (e.g., custom MoEGate). If required attributes are missing, skip with a note.
        """
        print(f"  Adjusting router: {old_num_experts} → {new_num_experts} outputs")

        # Ensure router exposes at least a weight matrix shaped [num_experts, in_features]
        if not hasattr(router, "weight"):
            print("  ⚠️ Router has no 'weight' attribute; skipping router adjustment.")
            return

        original_weight = router.weight.data
        in_features = getattr(router, "in_features", original_weight.shape[1])

        original_bias = None
        if hasattr(router, "bias") and getattr(router, "bias") is not None:
            original_bias = router.bias.data

        # Rebuild weight with removed expert rows
        new_weight = torch.zeros(
            new_num_experts,
            in_features,
            dtype=original_weight.dtype,
            device=original_weight.device,
        )

        new_idx = 0
        for i in range(old_num_experts):
            if i not in experts_to_remove:
                new_weight[new_idx] = original_weight[i]
                new_idx += 1

        # Assign back
        router.weight = nn.Parameter(new_weight)
        if hasattr(router, "out_features"):
            router.out_features = new_num_experts
        elif hasattr(router, "num_experts"):
            # Some custom gates track this field
            try:
                router.num_experts = new_num_experts
            except Exception:
                pass

        # Bias is optional
        if original_bias is not None and hasattr(router, "bias"):
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

        print("  ✓ Router adjusted successfully (weight" + (" + bias" if original_bias is not None else "") + ")")


def load_expert_counts_and_select_bottom_k(expert_counts_path: str, top_k: int) -> Dict[int, Set[int]]:
    """
    Load expert counts from a .pt file and select bottom-k experts per layer.
    
    Args:
        expert_counts_path: Path to expert_counts.pt file (tensor of shape [num_layers, num_experts])
        top_k: Number of least-used experts to remove per layer
    
    Returns:
        Dict mapping layer_idx -> set of expert indices to remove
    """
    import torch
    
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
        # print(f"   Layer {layer_idx}: Remove experts {experts_to_remove}")
        layer_specific_experts[layer_idx] = experts_to_remove
        
        # Show which experts will be removed
        removed_usages = [layer_counts[i].item() for i in experts_to_remove]
        print(f"   Layer {layer_idx}: Remove experts {experts_to_remove} (usage: {removed_usages})")
    
    return layer_specific_experts


def generate_expert_counts_via_tracing(model_name: str, dataset: str, save_dir: str) -> str:
    """
    Run tracing/moe_tracing.py to generate expert_counts for the given model and dataset,
    saving outputs into save_dir. Returns the expected path to the expert_counts.pt file.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tracer_path = os.path.join(project_root, "tracing", "moe_tracing.py")

    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        "python3", tracer_path,
        "--model_name", model_name,
        "--dataset", dataset,
        "--save_dir", save_dir,
    ]

    env = os.environ.copy()
    # Make project imports (e.g., utils) resolvable inside the tracer
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else project_root

    print("\n" + "-" * 60)
    print("Tracing experts to generate usage counts (auto)…")
    print(f"Model:   {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Save to: {save_dir}")
    print("-" * 60)

    try:
        res = subprocess.run(cmd, env=env, cwd=project_root, capture_output=True, text=True, check=True)
        # Echo concise tracer output tail for visibility
        tail = "\n".join(res.stdout.strip().splitlines()[-10:])
        if tail:
            print(tail)
    except subprocess.CalledProcessError as e:
        print("❌ Tracing failed:")
        if e.stdout:
            print("--- stdout ---\n" + e.stdout)
        if e.stderr:
            print("--- stderr ---\n" + e.stderr)
        raise

    base_name = model_name.split("/")[-1]
    counts_path = os.path.join(save_dir, f"{base_name}_expert_counts.pt")
    if not os.path.exists(counts_path):
        raise FileNotFoundError(f"Expected counts file not found: {counts_path}")

    print(f"✓ Found counts file: {counts_path}")
    return counts_path


def _normalize_eval_dataset_alias(name: str) -> str:
    """Map tracing dataset names to lm_eval task ids for run_lm_eval.py."""
    n = (name or "").lower()
    # Canonicalize common aliases to lm_eval tasks
    if n in {"wikitext2", "wikitext"}:
        return "wikitext"
    if n in {"gsm8k"}:
        return "gsm8k"
    # Map all WMT aliases to the lm_eval task id actually present: wmt16
    # Previously returned "wmt2016", which is missing in the task registry.
    if n in {"wmt16", "wmt", "wmt14", "wmt2016"}:
        return "wmt16"
    if n in {"humaneval"}:
        return "humaneval"
    # passthrough for known tasks
    if n in {"hellaswag", "piqa", "mmlu"}:
        return n
    # default: pass through raw name
    return n


def run_lm_eval_for_dataset(model_dir: str, dataset: Optional[str], limit: Optional[int] = None, num_fewshot: Optional[int] = None) -> None:
    """Run evaluation using lm_eval from external/MoE-Quantization for the saved model directory."""
    if not dataset:
        print("ℹ️ No --dataset provided; skipping evaluation.")
        return

    # Import lm_eval components (already in sys.path from top-level import)
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import initialize_tasks

    tasks = _normalize_eval_dataset_alias(dataset)
    
    # Set task-specific defaults for num_fewshot if not provided
    if num_fewshot is None:
        task_fewshot_defaults = {
            "gsm8k": 5,  # GSM8K needs few-shot for base models
            "humaneval": 0,
            "hellaswag": 0,
            "wikitext": 0,
            "mmlu": 5,
            "piqa": 0,
        }
        num_fewshot = task_fewshot_defaults.get(tasks, 0)
    
    print("\n" + "-" * 60)
    print("Running lm_eval on saved model…")
    print(f"Model dir: {model_dir}")
    print(f"Dataset:   {dataset}  → tasks={tasks}")
    print(f"Few-shot:  {num_fewshot} examples")
    if limit:
        print(f"Limit:     {limit} examples per task")
    print("-" * 60)

    try:
        # Set environment variables
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

        # Load tokenizer and model
        print(f"\n📥 Loading model from {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=False,
            trust_remote_code=True,
        )
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # Force cache and attention workarounds
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"

        print("✓ Model loaded successfully")

        # Initialize task registry
        initialize_tasks(verbosity="INFO")

        # Prepare task list
        task_list = [t.strip() for t in tasks.split(",") if t.strip()]
        
        print(f"\n📊 Tasks to evaluate: {', '.join(task_list)}\n")
        print(f"⚡ Starting evaluation...\n")

        all_metrics = {}
        results_per_task = {}

        for task_name in task_list:
            print(f"📋 Evaluating: {task_name}")
            
            try:
                lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8)

                results = evaluator.simple_evaluate(
                    model=lm,
                    tasks=task_name,
                    num_fewshot=num_fewshot,
                    batch_size=8,
                    log_samples=False,
                    limit=limit,
                )

                # Store results
                if task_name in results.get("results", {}):
                    task_res = results["results"][task_name]
                else:
                    task_res = next(iter(results.get("results", {}).values()), {})

                results_per_task[task_name] = task_res
                
                # Extract numeric metrics
                if isinstance(task_res, dict):
                    for key, value in task_res.items():
                        if isinstance(value, (int, float)):
                            all_metrics[f"{task_name}_{key}"] = value
                            print(f"   ✓ {key}: {value:.4f}")
                else:
                    print(f"   ✓ Saved full results for {task_name}")

            except Exception as e:
                print(f"   ❌ Error evaluating {task_name}: {e}")
                continue

        # Print summary
        print(f"\n📊 Evaluation complete!\n")
        print("Results:")
        print("-" * 50)
        for task_name, value in results_per_task.items():
            if isinstance(value, (int, float)):
                print(f"  {task_name:20s}: {value:.4f}")
            elif isinstance(value, dict):
                compact = json.dumps(value, ensure_ascii=False)[:200]
                print(f"  {task_name:20s}: {compact}")
        print("-" * 50)

        if all_metrics:
            average = sum(all_metrics.values()) / len(all_metrics)
            print(f"  {'average':20s}: {average:.4f}\n")

        # Save results to JSON
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_stub = os.path.basename(model_dir)
        results_file = os.path.join(model_dir, f"eval_result_{model_stub}_{timestamp}.json")

        output_data = {
            "model_path": model_dir,
            "dataset": dataset,
            "tasks": task_list,
            "metrics": all_metrics,
            "results_per_task": results_per_task,
            "timestamp": timestamp,
        }

        with open(results_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"💾 Results saved to: {results_file}\n")

    except Exception as e:
        print(f"❌ lm_eval failed: {e}")
        import traceback
        traceback.print_exc()
        return


def main():
    """Example usage with different modes"""
    
    parser = argparse.ArgumentParser(
        description="Remove least-used experts from MoE models based on usage statistics"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepseek-ai/deepseek-moe-16b-base",
        help="HuggingFace model name or path to local model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./moe_modified_efficient",
        help="Directory to save the modified model",
    )
    parser.add_argument(
        "--expert-counts",
        type=str,
        help="Path to expert_counts.pt file from tracing",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Number of least-used experts to remove per layer",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["usage-based", "manual"],
        default="manual",
        help="Mode: 'usage-based' uses expert counts, 'manual' uses hardcoded config",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name to run tracing on when auto-generating expert counts (e.g., wikitext2, gsm8k, hellaswag, wmt14, humaneval, ds1000, swebench, agentbench)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples per task for lm_eval (for quick tests)",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples for lm_eval (default: task-specific, typically 5 for gsm8k)",
    )
    
    args = parser.parse_args()
    
    MODEL_NAME = args.model_name
    OUTPUT_DIR = args.output_dir

    print("=" * 60)
    print("MoE Expert Removal - Memory Efficient Mode")
    print("=" * 60)

    # Determine which mode to use
    if args.mode == "usage-based":
        if args.top_k == None:
            print("❌ Error: --top-k is required for usage-based mode")
            return 1

        counts_path = args.expert_counts
        if not counts_path:
            # Auto-generate counts via tracing
            if not args.dataset:
                print("❌ Error: Provide --expert-counts or specify --dataset to auto-generate counts via tracing")
                return 1
            try:
                counts_path = generate_expert_counts_via_tracing(MODEL_NAME, args.dataset, OUTPUT_DIR)
            except Exception:
                return 1

        # Load expert counts and generate layer-specific removal
        layer_specific_experts = load_expert_counts_and_select_bottom_k(
            counts_path,
            args.top_k
        )
        
        print("\n" + "=" * 60)
        print("Usage-Based Expert Removal")
        print("=" * 60)
        
        modifier = MoEModifierMemoryEfficient(
            model_name=MODEL_NAME,
            layer_specific_experts=layer_specific_experts,
        )
        
    else:
        # Manual mode with hardcoded config
        print("\n" + "=" * 60)
        print("Manual Expert Removal (Hardcoded Config)")
        print("=" * 60)
        
        # All layers remove 2 experts (different ones, but same count!)
        modifier = MoEModifierMemoryEfficient(
            model_name=MODEL_NAME,
            layer_specific_experts={
                # All 28 transformer layers (0-27). Each removes 2 experts; counts stay consistent.
                0: [0, 1],   # L0
                1: [3, 5],   # L1
                2: [10, 15], # L2
                3: [2, 7],   # L3
                4: [0, 1],   # L4
                5: [3, 5],   # L5
                6: [10, 15], # L6
                7: [2, 7],   # L7
                8: [0, 1],   # L8
                9: [3, 5],   # L9
                10: [10, 15],# L10
                11: [2, 7],  # L11
                12: [0, 1],  # L12
                13: [3, 5],  # L13
                14: [10, 15],# L14
                15: [2, 7],  # L15
                16: [0, 1],  # L16
                17: [3, 5],  # L17
                18: [10, 15],# L18
                19: [2, 7],  # L19
                20: [0, 1],  # L20
                21: [3, 5],  # L21
                22: [10, 15],# L22
                23: [2, 7],  # L23
                24: [0, 1],  # L24
                25: [3, 5],  # L25
                26: [10, 15],# L26
                27: [2, 7],  # L27
            },
        )

    # ========================================
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
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # Avoid DynamicCache / SDPA paths that reference seen_tokens
        if hasattr(test_model.config, "use_cache"):
            test_model.config.use_cache = False
        if hasattr(test_model.config, "_attn_implementation"):
            test_model.config._attn_implementation = "eager"

        # Place model on a single device explicitly
        test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_model.to(test_device)

        print("✓ Model loaded successfully!")

        test_prompt = "The future of AI is"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(test_device)

        with torch.no_grad():
            outputs = test_model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )

        print(f"\nPrompt: {test_prompt}")
        print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        print("\n✅ Test successful! Model works perfectly with HuggingFace!")

    except Exception as e:
        print(f"⚠️ Test failed: {e}")
        import traceback

        traceback.print_exc()

    # print("\n🧹 Cleaning up...")
    # del model
    # if "test_model" in locals():
    #     del test_model
    # gc.collect()

    print("\n🎉 Done! Modified model saved to:", OUTPUT_DIR)

    # ------------------------------------------------------------------
    # Optional: run lm_eval on the saved model for the provided dataset
    # ------------------------------------------------------------------
    try:
        run_lm_eval_for_dataset(OUTPUT_DIR, args.dataset, args.limit, args.num_fewshot)
    except Exception as e:
        print(f"⚠️ Skipping evaluation due to error: {e}")


if __name__ == "__main__":
    main()