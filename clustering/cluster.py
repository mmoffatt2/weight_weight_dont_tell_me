import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import pdb

def main():
    usages, tok_count = parse_expert_usage("pruned_runs/pruned_moe_1expert_hellaswag/deepseek-moe-16b-base_token_trace.json")

    print("Token count: ",tok_count)
    total_usage = usages.sum(axis=0)
    normalized_usage = total_usage /tok_count.sum()

    clusters = cluster_batches(total_usage, 3)
    print(clusters.labels_)
    print(clusters.cluster_centers_)

    # fig, ax = plt.subplots(figsize=(8, 5))
    # plot_usage("Total Usage", normalized_usage, ax)
    # plt.show()


# num layers by num experts
def plot_usage(title, usage_2d, ax, label="Normalized Usage per Token"):
    """Plot usage heatmap of layers x experts"""
    
    # change to 1-indexing in display
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

# batches are single prompts currently
def parse_expert_usage(fname):
    """Parse filename into (batch x num_layers x num_experts)"""
    df = pd.read_json(fname)

    # can eventually pull this from config
    num_expert_layers = 27
    num_experts = 63
    batches = int(len(df) / 27)
    usage = np.zeros((batches,num_expert_layers,num_experts))
    tok_count = np.zeros(batches)

    for row in df.itertuples():
        batch_idx = row.batch_idx

        layer_idx = int(row.layer_name.split(".")[-3])

        # this is set multiple times
        tok_count[row.batch_idx] = len(row.topk_indices)

        # topk_indices is sequence length x k_experts
        for tok_topk in row.topk_indices:
            for expert_idx in tok_topk:
                # layer and expert idx are 1 indexed
                usage[batch_idx][layer_idx-1][expert_idx-1] += 1

    return usage, tok_count

def cluster_batches(usage, k=8):
    # return a list of clusters
    cluster_ids = np.zeros(usage.shape[0]) 
    x = usage.reshape((usage.shape[0],-1))
    return KMeans(n_clusters=k).fit(x)

if __name__ == "__main__":
    main()
