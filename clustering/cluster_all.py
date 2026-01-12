import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os
import multiprocessing
from pathlib import Path
import glob
import traceback
from datetime import datetime

MAX_WORKERS = 6

def parse_expert_usage(fname):
    df = pd.read_json(fname)
    num_expert_layers = 27
    num_experts = 63
    batches = int(len(df) / 27)
    usage = np.zeros((batches, num_expert_layers, num_experts))
    tok_count = np.zeros(batches)

    for row in df.itertuples():
        batch_idx = row.batch_idx
        layer_idx = int(row.layer_name.split(".")[-3])
        tok_count[batch_idx] = len(row.topk_indices)

        for tok_topk in row.topk_indices:
            for expert_idx in tok_topk:
                usage[batch_idx][layer_idx - 1][expert_idx - 1] += 1

    return usage, tok_count

def plot_usage(title, usage_2d, ax, label="Normalized Usage per Token"):
    sns.heatmap(
        usage_2d,
        cmap="YlGnBu",
        ax=ax,
        xticklabels=[f"E{j+1}" for j in range(usage_2d.shape[1])],
        yticklabels=[f"L{j+1}" for j in range(usage_2d.shape[0])],
        cbar_kws={"label": label},
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Expert ID", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)

def process_model(json_path, output_dir):
    model_name = Path(json_path).parent.name
    print(f"[{model_name}] Loading data...", flush=True)

    usages, tok_count = parse_expert_usage(json_path)
    total_usage = usages.sum(axis=0)
    x = usages.reshape((usages.shape[0], -1))

    fig_dir = Path(output_dir) / model_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [f"Clustering Report for {model_name}", f"Total batches: {usages.shape[0]}\n"]

    print(f"[{model_name}] Running KMeans k=2-10...", flush=True)
    for k in range(2, 7):
        clusters = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300).fit(x)
        counts = np.bincount(clusters.labels_, minlength=k)
        total = len(clusters.labels_)
        dist_str = ", ".join([f"Cluster {i}: {c} ({c/total:.1%})" for i, c in enumerate(counts)])
        report_lines.append(f"k={k}: {dist_str}")

    (fig_dir / "clustering_report.txt").write_text("\n".join(report_lines))

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_usage("Overall Usage", total_usage / tok_count.sum(), ax)
    fig.savefig(fig_dir / "overall_usage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Generate cluster visualizations for k=3,4,5,6
    for k in [3, 4, 5, 6]:
        print(f"[{model_name}] Generating k={k} visualizations...", flush=True)
        
        kmeans = KMeans(n_clusters=k, random_state=42).fit(x)
        
        # Calculate grid layout
        if k == 3:
            rows, cols = 3, 1
        elif k == 4:
            rows, cols = 2, 2
        else:  # k=5 or k=6
            rows, cols = 3, 2
        
        # Create grid of cluster heatmaps
        fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 5*rows))
        axes = axes.flatten()
        
        for i in range(k):
            center = kmeans.cluster_centers_[i].reshape(27, 63)
            plot_usage(f"Cluster {i} Pattern (n={np.sum(kmeans.labels_ == i)})", center, axes[i], label="Expert Activation")
        
        # Hide unused subplots
        for i in range(k, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        fig.savefig(fig_dir / f"clusters_k{k}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        # Create pie chart for k clusters
        counts = np.bincount(kmeans.labels_, minlength=k)
        labels = [f"Cluster {i} ({counts[i]}, {counts[i]/len(kmeans.labels_):.1%})" for i in range(k)]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title(f"Cluster Distribution for {model_name} (k={k})", fontsize=14)
        fig.savefig(fig_dir / f"cluster_k{k}_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

def process_model_wrapper(args):
    """Wrapper for parallel execution with progress tracking"""
    json_path, output_dir, progress_file = args
    model_name = Path(json_path).parent.name
    
    try:
        process_model(json_path, output_dir)
        
        # Log success
        with open(progress_file, 'a') as f:
            f.write(f"[SUCCESS] {model_name}\n")
        
        return (model_name, True, None)
    except Exception as e:
        # Log error with full stack trace
        error_traceback = traceback.format_exc()
        with open(progress_file, 'a') as f:
            f.write(f"[ERROR] {model_name}: {str(e)}\n{error_traceback}\n")
        
        return (model_name, False, error_traceback)

def main():
    json_files = glob.glob("pruned_models/**/*token_trace.json", recursive=True)
    output_dir = Path("plots")
    
    # Clean up previous results
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    
    # Configure parallel processing
    n_workers = min(MAX_WORKERS, len(json_files), multiprocessing.cpu_count())
    print(f"Processing {len(json_files)} models using {n_workers} workers...")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Set up progress tracking
    progress_file = "cluster_progress.log"
    with open(progress_file, 'w') as f:
        f.write(f"Starting processing of {len(json_files)} models at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Using {n_workers} workers\n\n")
    
    # Prepare arguments for each worker
    args_list = [(f, output_dir, progress_file) for f in json_files]
    
    # Process in parallel with real-time progress
    start_time = datetime.now()
    completed = 0
    errors = []
    
    with multiprocessing.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_model_wrapper, args_list):
            completed += 1
            model_name, success, error = result
            
            if success:
                print(f"[{completed}/{len(json_files)}] ✓ {model_name}", flush=True)
            else:
                errors.append((model_name, error))
                print(f"[{completed}/{len(json_files)}] ✗ {model_name}", flush=True)
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"Completed: {completed}/{len(json_files)} models")
    print(f"Errors: {len(errors)}")
    print(f"Duration: {duration}")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if errors:
        print(f"\n{'='*60}")
        print(f"Failed models ({len(errors)}):")
        for model_name, error in errors[:10]:
            print(f"  - {model_name}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        print(f"\nCheck {progress_file} for full error details")
    
    print(f"\nResults saved to {output_dir}/")
    print(f"Progress log saved to {progress_file}")

if __name__ == "__main__":
    main()
