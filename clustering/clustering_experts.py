import json
import glob
import numpy as np
import pdb
from collections import Counter
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns


# various distance matris for clustering
def hellinger_distance(freq1, freq2):
    """
    Hellinger distance - excellent for probability/frequency distributions.
    Range: [0, 1], where 0 = identical, 1 = completely different.
    """
    # Normalize to probabilities
    p1 = freq1.flatten() / (freq1.sum() + 1e-10)
    p2 = freq2.flatten() / (freq2.sum() + 1e-10)
    return np.sqrt(np.sum((np.sqrt(p1) - np.sqrt(p2)) ** 2)) / np.sqrt(2)


def chi_squared_distance(freq1, freq2, epsilon=1e-10):
    """
    Chi-squared distance for frequency distributions.
    Handles the statistical nature of count data.
    """
    freq1 = freq1.flatten()
    freq2 = freq2.flatten()
    # Add small epsilon to avoid division by zero
    sum_freq = freq1 + freq2 + epsilon
    return np.sqrt(np.sum((freq1 - freq2) ** 2 / sum_freq))


def jensen_shannon_distance(freq1, freq2):
    """
    Jensen-Shannon distance - symmetric version of KL divergence.
    Good for comparing probability distributions.
    """
    from scipy.spatial.distance import jensenshannon

    p1 = freq1.flatten() / (freq1.sum() + 1e-10)
    p2 = freq2.flatten() / (freq2.sum() + 1e-10)
    return jensenshannon(p1, p2)


def compute_frequency_distance_matrix(tensors, metric="hellinger"):
    """
    Compute pairwise distances for frequency data.
    """
    n = len(tensors)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = hellinger_distance(tensors[i], tensors[j])
            distances[i, j] = d
            distances[j, i] = d

    return distances


def cluster_batches(batch_activations, k, method="hellinger"):
    """
    Cluster batches based on their expert usage patterns.

    Parameters:
    -----------
    batch_activations : list of np.ndarray
        List of 2D arrays (num_layers x num_experts) per batch
    k : int
        Number of clusters
    method : str
        'hellinger' or 'cosine'

    Returns:
    --------
    labels : np.ndarray
        Cluster assignment for each batch
    centers : np.ndarray
        Cluster centers
    distances : np.ndarray
        Pairwise distance matrix
    """
    n_batches = len(batch_activations)

    if method == "cosine":
        # Cosine similarity works well for frequency patterns
        flattened = np.array([t.flatten() for t in batch_activations])
        # Normalize by total (convert to proportions)
        flattened = flattened / (flattened.sum(axis=1, keepdims=True) + 1e-10)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(flattened)
        centers = kmeans.cluster_centers_.reshape(k, *batch_activations[0].shape)
        distances = None

    else:  # hellinger
        # Use custom distance metric with hierarchical clustering
        distances = compute_frequency_distance_matrix(batch_activations, method)

        # Convert to condensed form for linkage
        condensed = squareform(distances)

        # Perform hierarchical clustering
        Z = linkage(condensed, method="average")
        labels = fcluster(Z, k, criterion="maxclust") - 1

        # Compute cluster centers (mean of members)
        tensor_shape = batch_activations[0].shape
        centers = np.zeros((k, *tensor_shape))
        for i in range(k):
            cluster_members = [
                batch_activations[j] for j in range(n_batches) if labels[j] == i
            ]
            if cluster_members:
                centers[i] = np.mean(cluster_members, axis=0)

    return labels, centers, distances


def save_cluster_partitions(
    labels, batch_indices, centers, data_name, prefix, method, k
):
    """
    Save cluster partitions to a JSON file.

    Parameters:
    -----------
    labels : np.ndarray
        Cluster assignments for each batch
    batch_indices : list
        Original batch indices
    centers : np.ndarray
        Cluster centers (2D arrays)
    data_name : str
        Dataset name
    prefix : str
        Output directory prefix
    method : str
        Clustering method used
    k : int
        Number of clusters
    """
    # Build the partition data structure
    partition_data = {
        "metadata": {
            "dataset": data_name,
            "method": method,
            "n_clusters": k,
            "n_batches": len(batch_indices),
            "n_layers": centers.shape[1] if len(centers) > 0 else 0,
            "n_experts": centers.shape[2] if len(centers) > 0 else 0,
        },
        "clusters": [],
    }

    # Organize batches by cluster
    for cluster_id in range(k):
        cluster_batch_ids = [
            int(batch_indices[i])
            for i in range(len(batch_indices))
            if labels[i] == cluster_id
        ]

        cluster_info = {
            "cluster_id": int(cluster_id),
            "size": len(cluster_batch_ids),
            "batch_indices": cluster_batch_ids,
            "center": centers[cluster_id].tolist(),  # Convert numpy to list
        }

        partition_data["clusters"].append(cluster_info)

    # Save to JSON file
    output_file = f"{prefix}/{data_name}/cluster_partitions_{method}_k{k}.json"
    with open(output_file, "w") as f:
        json.dump(partition_data, f, indent=2)

    print(f"  Saved partitions to {output_file}")

    return output_file


def save_all_k_partitions(
    batch_activations, batch_indices, k_values, data_name, prefix, method
):
    """
    Save partitions for all k values tested.

    Returns a summary JSON with paths to all partition files.
    """
    summary = {
        "dataset": data_name,
        "method": method,
        "n_batches": len(batch_indices),
        "k_values": [],
    }

    for k in k_values:
        if k >= len(batch_activations):
            continue

        labels, centers, _ = cluster_batches(batch_activations, k=k, method=method)

        # Save this k's partition
        partition_file = save_cluster_partitions(
            labels, batch_indices, centers, data_name, prefix, method, k
        )

        # Calculate cluster statistics
        cluster_sizes = [int(np.sum(labels == i)) for i in range(k)]

        summary["k_values"].append(
            {
                "k": k,
                "partition_file": partition_file,
                "cluster_sizes": cluster_sizes,
            }
        )

    # Save summary
    summary_file = f"{prefix}/{data_name}/cluster_summary_{method}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved clustering summary to {summary_file}")

    return summary


def plot_cluster_centers(centers, data_name, prefix, method="hellinger"):
    """
    Visualize cluster centers as heatmaps.
    """
    k = len(centers)
    fig, axes = plt.subplots(1, k, figsize=(6 * k, 6))
    if k == 1:
        axes = [axes]

    for i in range(k):
        sns.heatmap(
            centers[i],
            cmap="YlOrRd",
            ax=axes[i],
            xticklabels=[f"E{j}" for j in range(centers[i].shape[1])],
            yticklabels=[f"L{j}" for j in range(centers[i].shape[0])],
            cbar_kws={"label": "Normalized Usage"},
        )
        axes[i].set_title(f"Cluster {i} Center", fontsize=14)
        axes[i].set_xlabel("Expert ID", fontsize=12)
        axes[i].set_ylabel("Layer", fontsize=12)

    plt.tight_layout()
    fig.savefig(f"{prefix}/{data_name}/batch_clusters_{method}.png", dpi=300)
    plt.close()


def plot_batch_assignments(
    labels, batch_activations, data_name, prefix, method="hellinger"
):
    """
    Plot which batches belong to which cluster.
    """
    n_clusters = len(np.unique(labels))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a visual representation
    batch_indices = np.arange(len(labels))
    colors = plt.cm.tab10(labels)

    ax.scatter(batch_indices, labels, c=colors, s=100, alpha=0.6)
    ax.set_xlabel("Batch Index", fontsize=12)
    ax.set_ylabel("Cluster Assignment", fontsize=12)
    ax.set_title(f"{data_name} Batch Clustering ({method})", fontsize=14)
    ax.set_yticks(range(n_clusters))
    ax.grid(True, alpha=0.3)

    # Add cluster size annotations
    for i in range(n_clusters):
        count = np.sum(labels == i)
        ax.text(len(labels) + 1, i, f"n={count}", fontsize=10, va="center")

    plt.tight_layout()
    fig.savefig(f"{prefix}/{data_name}/batch_assignments_{method}.png", dpi=300)
    plt.close()


def plot_distance_matrix(distances, labels, data_name, prefix, method="hellinger"):
    """
    Plot the pairwise distance matrix between batches, ordered by cluster.
    """
    if distances is None:
        return

    # Sort by cluster labels
    sorted_indices = np.argsort(labels)
    sorted_distances = distances[sorted_indices][:, sorted_indices]
    sorted_labels = labels[sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        sorted_distances,
        cmap="viridis",
        ax=ax,
        square=True,
        cbar_kws={"label": "Hellinger Distance"},
    )

    # Add cluster boundaries
    boundaries = [0]
    for i in range(len(np.unique(labels))):
        boundaries.append(boundaries[-1] + np.sum(sorted_labels == i))

    for b in boundaries[1:-1]:
        ax.axhline(b, color="red", linewidth=2)
        ax.axvline(b, color="red", linewidth=2)

    ax.set_title(f"{data_name} Batch Distance Matrix ({method})", fontsize=14)
    ax.set_xlabel("Batch Index (sorted by cluster)", fontsize=12)
    ax.set_ylabel("Batch Index (sorted by cluster)", fontsize=12)

    plt.tight_layout()
    fig.savefig(f"{prefix}/{data_name}/batch_distance_matrix_{method}.png", dpi=300)
    plt.close()


def main():
    prefix = "./quantized_runs/8bit_quant/"
    for data_name in [
        "deepseek-moe-16b-base_gsm8k_100experts/",
        "deepseek-moe-16b-base_gsm8k_70experts/",
        "deepseek-moe-16b-base_gsm8k_allexperts/",
        "deepseek-moe-16b-base_gsm8k_noexperts/",
        "deepseek-moe-16b-base_hellaswag_allexperts/",
        "deepseek-moe-16b-base_hellaswag_noexperts/",
        "deepseek-moe-16b-base_wmt16_100experts/",
        "deepseek-moe-16b-base_wmt16_50experts/",
        "deepseek-moe-16b-base_wmt16_70experts/",
        "deepseek-moe-16b-base_wmt16_noexperts/",
    ]:
        # Find the json with the shortest name ending in token_trace.json
        best_fname = None
        for fname in glob.glob(f"{prefix}/{data_name}/*.json"):
            if not fname.endswith("token_trace.json"):
                continue
            if best_fname is None or len(fname) < len(best_fname):
                best_fname = fname

        fname = best_fname

        # Read in json from file
        data = None
        print(f"Reading in {fname}, found in {prefix}/{data_name}")
        with open(fname, "r") as f:
            data = json.load(f)

        max_index = 0
        layer_ctrs = {}
        batch_layer_usage = {}  # NEW: store per-batch, per-layer usage

        for elem in data:
            sample_idx = elem["batch_idx"]
            layer_name = elem["layer_name"]

            use_ctr = Counter()
            for experts in elem["topk_indices"]:
                max_index = max(max_index, max(experts))
                use_ctr.update(experts)

            # Store per-batch usage
            if sample_idx not in batch_layer_usage:
                batch_layer_usage[sample_idx] = {}

            normalized_ctr = {}
            for idx in use_ctr:
                # normalize by number of tokens in batch
                normalized_ctr[idx] = use_ctr[idx] / len(elem["topk_indices"])

            batch_layer_usage[sample_idx][layer_name] = normalized_ctr

            if layer_name not in layer_ctrs:
                layer_ctrs[layer_name] = []

            layer_ctrs[layer_name].append(normalized_ctr)

        print(f"max index {max_index}")

        # Calculate variances between batches
        std_per_layer = {}
        mean_per_layer = {}
        sum_per_layer = {}

        for layer_name in layer_ctrs:
            print(layer_name)

            layer_std = []
            layer_mean = []
            layer_sum = []
            for idx in range(max_index + 1):
                layer_val = []
                for ctr in layer_ctrs[layer_name]:
                    if idx not in ctr:
                        layer_val.append(0)
                    else:
                        layer_val.append(ctr[idx])

                layer_std.append(np.std(layer_val))
                layer_mean.append(np.mean(layer_val))
                layer_sum.append(np.sum(layer_val))

            layer_std = np.array(layer_std)
            layer_mean = np.array(layer_mean)
            std_per_layer[layer_name] = layer_std
            mean_per_layer[layer_name] = layer_mean
            sum_per_layer[layer_name] = layer_sum

            print("layer_mean", layer_mean)
            print("layer_std", layer_std)

        mean_2d = []
        std_2d = []
        sum_2d = []
        for layer_name in mean_per_layer:
            mean_2d.append(mean_per_layer[layer_name])
            std_2d.append(std_per_layer[layer_name])
            sum_2d.append(sum_per_layer[layer_name])
        mean_2d = np.array(mean_2d)
        std_2d = np.array(std_2d)
        sum_2d = np.array(sum_2d)

        coef_variation = mean_2d / (std_2d + 1e-10)

        # Plot original visualizations
        sns.set()

        fig = plt.figure(figsize=(12, 8))
        sns.heatmap(
            coef_variation,
            cmap="YlGnBu",
            xticklabels=[f"E{i}" for i in range(coef_variation.shape[1])],
            yticklabels=[f"L{i}" for i in range(coef_variation.shape[0])],
            cbar_kws={"label": "Coefficient of Variation in routing"},
        )
        plt.title(f"{data_name} CV routing Original", fontsize=16, pad=14)
        plt.xlabel("Expert ID", fontsize=13)
        plt.ylabel("Layer", fontsize=13)
        plt.tight_layout()
        fig.savefig(f"{prefix}/{data_name}/expert_coef_variation_orig.png", dpi=300)
        plt.close()

        fig = plt.figure(figsize=(12, 8))
        sns.heatmap(
            sum_2d,
            cmap="YlGnBu",
            xticklabels=[f"E{i}" for i in range(sum_2d.shape[1])],
            yticklabels=[f"L{i}" for i in range(sum_2d.shape[0])],
            cbar_kws={"label": ""},
        )
        plt.title(f"{data_name} Total Usage Original", fontsize=16, pad=14)
        plt.xlabel("Expert ID", fontsize=13)
        plt.ylabel("Layer", fontsize=13)
        plt.tight_layout()
        fig.savefig(f"{prefix}/{data_name}/expert_usage_orig.png", dpi=300)
        plt.close()

        # ===== NEW: CLUSTERING ANALYSIS =====
        print("\n" + "=" * 60)
        print(f"CLUSTERING BATCHES FOR {data_name}")
        print("=" * 60)

        # Build 3D tensor: [num_batches, num_layers, num_experts]
        batch_indices = sorted(batch_layer_usage.keys())
        layer_names = sorted(layer_ctrs.keys())
        n_batches = len(batch_indices)
        n_layers = len(layer_names)
        n_experts = max_index + 1

        # Create per-batch activation matrices
        batch_activations = []
        for batch_idx in batch_indices:
            batch_matrix = np.zeros((n_layers, n_experts))
            for layer_idx, layer_name in enumerate(layer_names):
                if layer_name in batch_layer_usage[batch_idx]:
                    for expert_idx, usage in batch_layer_usage[batch_idx][
                        layer_name
                    ].items():
                        batch_matrix[layer_idx, expert_idx] = usage
            batch_activations.append(batch_matrix)

        print(f"Created {n_batches} batch activation matrices")
        print(f"Each matrix shape: ({n_layers} layers, {n_experts} experts)")

        # Determine optimal number of clusters (try multiple k values)
        k_values = [2, 3, 4, 5]
        best_k = 3  # default

        # Try to find reasonable k based on number of batches
        if n_batches < 10:
            k_values = [2, 3]
            best_k = 2
        elif n_batches < 20:
            k_values = [2, 3, 4]
            best_k = 3

        for method in ["hellinger", "cosine"]:
            print(f"\n--- Clustering with {method.upper()} distance ---")

            # Save partitions for all k values
            print(f"Saving partitions for all k values...")
            summary = save_all_k_partitions(
                batch_activations, batch_indices, k_values, data_name, prefix, method
            )

            for k in k_values:
                if k >= n_batches:
                    continue

                labels, centers, distances = cluster_batches(
                    batch_activations, k=k, method=method
                )

                cluster_sizes = [np.sum(labels == i) for i in range(k)]
                print(f"k={k}: Cluster sizes: {cluster_sizes}")

            # Use best_k for final visualization
            k = min(best_k, n_batches - 1)
            labels, centers, distances = cluster_batches(
                batch_activations, k=k, method=method
            )

            print(f"\nFinal clustering (k={k}):")
            for i in range(k):
                batch_ids = [
                    batch_indices[j] for j in range(n_batches) if labels[j] == i
                ]
                print(
                    f"  Cluster {i}: {len(batch_ids)} batches - {batch_ids[:10]}{'...' if len(batch_ids) > 10 else ''}"
                )

            # Generate visualizations
            plot_cluster_centers(centers, data_name, prefix, method)
            plot_batch_assignments(labels, batch_activations, data_name, prefix, method)
            if distances is not None:
                plot_distance_matrix(distances, labels, data_name, prefix, method)

            print(f"Saved clustering visualizations for {method}")


if __name__ == "__main__":
    main()
