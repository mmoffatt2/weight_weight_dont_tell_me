#!/usr/bin/env python3
"""
Batch MoE Tracing Script
------------------------
Automatically discovers models in the runs/ folder and runs moe_tracing.py
on each one across multiple datasets.

Usage:
    python3 trace_all_models.py --datasets gsm8k hellaswag wmt16 --nsamples 64 --seqlen 512
    
    # Dry run to see what would be executed
    python3 trace_all_models.py --dry-run
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Set


def load_model_metadata(config_path: str = "configs/moe_model_metadata.json") -> Dict:
    """Load the model metadata configuration"""
    with open(config_path) as f:
        return json.load(f)


def discover_models_in_runs(runs_dir: str = "runs") -> List[Dict[str, str]]:
    """
    Discover models in the runs directory.
    
    Returns a list of dicts with:
        - path: full path to model dir (or quantized subfolder if it exists)
        - name: model directory name
        - base_model: the base HF model name (extracted from config if available)
        - dataset: dataset name extracted from folder name (or None)
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        print(f"❌ Runs directory not found: {runs_dir}")
        return []
    
    models = []
    
    for item in sorted(runs_path.iterdir()):
        if not item.is_dir():
            continue
        
        # Check if there's a quantized subfolder
        quantized_path = item / "quantized"
        if quantized_path.exists() and quantized_path.is_dir():
            model_path = quantized_path
            print(f"🔍 Found quantized subfolder for {item.name}")
        else:
            model_path = item
        
        # Try to load config.json to get the original model name
        config_file = model_path / "config.json"
        base_model = None
        
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    # Common fields that might contain the base model
                    base_model = config.get("_name_or_path") or config.get("model_type")
                    # Validate that base_model looks like a HuggingFace ID (org/model)
                    if base_model and "/" not in base_model:
                        print(f"⚠️  Invalid base model '{base_model}' in config, will infer from folder name")
                        base_model = None
            except Exception as e:
                print(f"⚠️  Could not read config from {item.name}: {e}")
        
        # Fallback: try to infer from directory name
        dir_name = item.name
        if not base_model:
            # Extract base model from directory name (e.g., "deepseek-moe-16b-base_gsm8k_50experts" -> "deepseek-ai/deepseek-moe-16b-base")
            if dir_name.startswith("deepseek-moe-16b"):
                base_model = "deepseek-ai/deepseek-moe-16b-base"
            elif dir_name.startswith("DeepSeek-V2-Lite"):
                base_model = "deepseek-ai/DeepSeek-V2-Lite"
            elif dir_name.startswith("Mixtral-8x7B"):
                base_model = "mistralai/Mixtral-8x7B-v0.1"
            elif dir_name.startswith("Qwen1.5-MoE"):
                base_model = "Qwen/Qwen1.5-MoE-A2.7B"
            else:
                print(f"⚠️  Could not infer base model for {dir_name}, skipping")
                continue
        
        # Extract dataset from folder name
        # Look for common dataset patterns: _gsm8k_, _hellaswag_, _wmt16_, etc.
        dataset = None
        dir_lower = dir_name.lower()
        known_datasets = ["gsm8k", "hellaswag", "wmt16", "wmt14", "humaneval", "ds1000", "swebench", "agentbench", "wikitext", "wikitext2"]
        
        for ds in known_datasets:
            if f"_{ds}_" in dir_lower or dir_lower.endswith(f"_{ds}"):
                dataset = ds
                break
        
        models.append({
            "path": str(model_path),
            "name": item.name,
            "base_model": base_model,
            "dataset": dataset,  # None if no dataset found in name
        })
    
    return models


def check_if_trace_exists(model_path: str, model_name: str, dataset: str) -> bool:
    """Check if tracing output already exists for this model + dataset combo"""
    # If model_path ends with 'quantized', check in the parent directory
    model_path_obj = Path(model_path)
    if model_path_obj.name == "quantized":
        check_dir = model_path_obj.parent
        # Add quantized prefix to filename
        trace_file = check_dir / f"quantized_{model_name}_{dataset}_expert_counts.pt"
    else:
        check_dir = model_path_obj
        trace_file = check_dir / f"{model_name}_{dataset}_expert_counts.pt"
    
    return trace_file.exists()


def run_tracing(
    model_path: str,
    model_name: str,
    base_model: str,
    dataset: str,
    config_path: str,
    nsamples: int,
    seqlen: int,
    batch_size: int,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> bool:
    """
    Run moe_tracing.py for a single model + dataset combination.
    
    Returns True if successful (or skipped), False on error.
    """
    
    # Check if already traced
    if skip_existing and check_if_trace_exists(model_path, model_name, dataset):
        print(f"  ⏭️  Trace already exists for {model_name} on {dataset}, skipping")
        return True
    
    # Convert all paths to absolute paths
    script_dir = Path(__file__).parent.absolute()
    abs_model_path = Path(model_path).absolute()
    abs_config_path = (script_dir / config_path).absolute()
    
    # If model_path ends with 'quantized', save results in the parent directory
    # Otherwise save in the model directory itself
    if abs_model_path.name == "quantized":
        abs_save_dir = abs_model_path.parent
    else:
        abs_save_dir = abs_model_path
    
    # Build command: use base_model for metadata lookup, but save to actual model path
    # model_path will be used to load the model and auto-derive the output filename
    cmd = [
        "python3",
        "tracing/moe_tracing.py",
        "--model_name", base_model,  # Use base model name for metadata lookup
        "--model_path", str(abs_model_path),  # Load from the actual model directory (quantized or not)
        "--config_path", str(abs_config_path),
        "--dataset", dataset,
        "--save_dir", str(abs_save_dir),  # Save results in the parent directory (not inside quantized/)
        "--nsamples", str(nsamples),
        "--seqlen", str(seqlen),
        "--batch_size", str(batch_size),
    ]
    
    print(f"\n{'=' * 80}")
    print(f"🚀 Tracing: {model_name} on {dataset}")
    print(f"{'=' * 80}")
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("  [DRY RUN - command not executed]")
        return True
    
    try:
        # Run the command from the script's directory (project root)
        # Add project root to PYTHONPATH so imports work
        env = os.environ.copy()
        env['PYTHONPATH'] = str(script_dir) + ':' + env.get('PYTHONPATH', '')
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
            cwd=script_dir,  # Run from project root
            env=env,  # Pass updated environment
        )
        print(f"✅ Successfully traced {model_name} on {dataset}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error tracing {model_name} on {dataset}")
        print(f"   Exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch trace all models in runs/ folder across multiple datasets"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory containing model folders (default: runs)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/moe_model_metadata.json",
        help="Path to model metadata config (default: configs/moe_model_metadata.json)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Datasets to trace on. If not specified, uses dataset from folder name automatically.",
    )
    parser.add_argument(
        "--auto-dataset",
        action="store_true",
        default=True,
        help="Automatically use the dataset extracted from each folder name (default: True)",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=64,
        help="Number of samples per dataset (default: 64)",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="Sequence length (default: 2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run tracing even if output already exists",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Only trace specific models (by directory name). If not set, traces all.",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 MoE Batch Tracing Script")
    print("=" * 80)
    
    # Discover models
    print(f"\n📂 Discovering models in {args.runs_dir}/...")
    all_models = discover_models_in_runs(args.runs_dir)
    
    if not all_models:
        print("❌ No models found in runs directory")
        return 1
    
    print(f"✅ Found {len(all_models)} models:")
    for m in all_models:
        dataset_info = f" | dataset: {m['dataset']}" if m['dataset'] else " | dataset: not detected"
        print(f"   • {m['name']} (base: {m['base_model']}{dataset_info})")
    
    # Filter if specific models requested
    if args.models:
        filter_set = set(args.models)
        all_models = [m for m in all_models if m['name'] in filter_set]
        print(f"\n🔍 Filtering to {len(all_models)} requested models")
    
    # Load metadata to validate models
    print(f"\n📋 Loading model metadata from {args.config_path}...")
    try:
        metadata = load_model_metadata(args.config_path)
        print(f"✅ Loaded metadata for {len(metadata)} model types")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return 1
    
    # Validate that all base models are in metadata
    missing = []
    for m in all_models:
        if m['base_model'] not in metadata:
            missing.append(m['base_model'])
    
    if missing:
        print(f"\n⚠️  Warning: {len(missing)} base models not found in metadata:")
        for base in set(missing):
            print(f"   • {base}")
        print("These models will be skipped.")
        all_models = [m for m in all_models if m['base_model'] not in missing]
    
    if not all_models:
        print("❌ No valid models to trace after filtering")
        return 1
    
    # Summary
    # Auto-detect dataset if --datasets not explicitly provided
    use_auto_dataset = args.auto_dataset or args.datasets is None
    
    if use_auto_dataset:
        # Count how many models have datasets
        models_with_dataset = [m for m in all_models if m['dataset']]
        total_jobs = len(models_with_dataset)
        dataset_summary = "auto (from folder names)"
    else:
        total_jobs = len(all_models) * len(args.datasets)
        dataset_summary = f"{len(args.datasets)} ({', '.join(args.datasets)})"
    
    print(f"\n{'=' * 80}")
    print(f"📊 EXECUTION PLAN")
    print(f"{'=' * 80}")
    print(f"Models:   {len(all_models)}")
    print(f"Datasets: {dataset_summary}")
    print(f"Total jobs: {total_jobs}")
    print(f"Samples per dataset: {args.nsamples}")
    print(f"Sequence length: {args.seqlen}")
    print(f"Batch size: {args.batch_size}")
    print(f"Skip existing: {not args.no_skip_existing}")
    print(f"Dry run: {args.dry_run}")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - no commands will be executed")
    
    # Execute tracing jobs
    print(f"\n{'=' * 80}")
    print("🚀 Starting batch tracing...")
    print(f"{'=' * 80}")
    
    results = {
        "success": [],
        "failed": [],
        "skipped": [],
    }
    
    for model in all_models:
        # Determine which datasets to trace for this model
        # Auto-detect dataset if --datasets not explicitly provided
        use_auto_dataset = args.auto_dataset or args.datasets is None
        
        if use_auto_dataset:
            # Use dataset from folder name
            if model['dataset']:
                datasets_to_trace = [model['dataset']]
            else:
                print(f"\n⚠️  Warning: {model['name']} has no dataset in folder name, skipping")
                continue
        else:
            # Use datasets from command line
            datasets_to_trace = args.datasets
        
        for dataset in datasets_to_trace:
            job_id = f"{model['name']}_{dataset}"
            
            # Check if exists
            if not args.no_skip_existing and check_if_trace_exists(model['path'], model['name'], dataset):
                print(f"\n⏭️  Skipping {job_id} (already traced)")
                results["skipped"].append(job_id)
                continue
            
            success = run_tracing(
                model_path=model['path'],
                model_name=model['name'],
                base_model=model['base_model'],
                dataset=dataset,
                config_path=args.config_path,
                nsamples=args.nsamples,
                seqlen=args.seqlen,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                skip_existing=not args.no_skip_existing,
            )
            
            if success:
                results["success"].append(job_id)
            else:
                results["failed"].append(job_id)
    
    # Summary
    print(f"\n{'=' * 80}")
    print("📊 BATCH TRACING SUMMARY")
    print(f"{'=' * 80}")
    print(f"✅ Successful: {len(results['success'])}")
    print(f"⏭️  Skipped:    {len(results['skipped'])}")
    print(f"❌ Failed:     {len(results['failed'])}")
    
    if results['failed']:
        print("\n❌ Failed jobs:")
        for job in results['failed']:
            print(f"   • {job}")
        return 1
    
    print("\n🎉 All tracing jobs completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
