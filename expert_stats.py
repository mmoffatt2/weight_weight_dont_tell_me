import json
import numpy as np
import pdb
from collections import Counter

#  save the dimensions [num batch, num layers, prompt len, topk per token]


def main():
    prefix = "quantized_runs"
    for data_name in [
        "deepseek-moe-16b-base_gsm8k_100experts",
        "deepseek-moe-16b-base_gsm8k_50experts",
        "deepseek-moe-16b-base_gsm8k_70experts",
        "deepseek-moe-16b-base_gsm8k_noexperts",
        "deepseek-moe-16b-base_wmt16_100experts",
        "deepseek-moe-16b-base_wmt16_50experts",
        "deepseek-moe-16b-base_wmt16_70experts",
        "deepseek-moe-16b-base_wmt16_noexperts",
    ]:
        # data_name = "deepseek_ds1000_no_padding"
        fname = f"{prefix}/{data_name}/deepseek-moe-16b-base_token_trace.json"

        # read in json from file
        data = None
        with open(fname, "r") as f:
            data = json.load(f)

        # array of dictionaries of batch_idx, layer_name, topk_indices, topk_values
        # batch_idx == sample_idx

        # I care about per batch statistics, and how they differ from the other batches

        # within each batch and layer, I get a list of top k indices for the given layer

        # elem is a dictionary of batch_idx, layer_name, topk_indices, topk_values

        max_index = 0
        layer_ctrs = {}

        for elem in data:
            sample_idx = elem["batch_idx"]
            layer_name = elem["layer_name"]

            use_ctr = Counter()
            for experts in elem["topk_indices"]:
                max_index = max(max_index, max(experts))
                use_ctr.update(experts)
            if layer_name not in layer_ctrs:
                layer_ctrs[layer_name] = []
            else:
                normalized_ctr = {}
                for idx in use_ctr:
                    # normalize by number of tokens in batch
                    normalized_ctr[idx] = use_ctr[idx] / len(elem["topk_indices"])

                layer_ctrs[layer_name].append(normalized_ctr)

        print(f"max index {max_index}")

        # calculate variances between batches
        std_per_layer = {}
        mean_per_layer = {}
        sum_per_layer = {}

        for layer_name in layer_ctrs:
            print(layer_name)
            # print(len(layer_ctrs[layer_name]))

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

        coef_variation = mean_2d / std_2d

        # plot in a 2d heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set()

        fig = plt.figure(figsize=(12, 8))  # larger figure
        sns.heatmap(
            coef_variation,
            cmap="YlGnBu",
            xticklabels=[f"E{i}" for i in range(coef_variation.shape[1])],
            yticklabels=[f"L{i}" for i in range(coef_variation.shape[0])],
            cbar_kws={"label": "Coefficient of Variation in routing"},
        )
        plt.title(f"{data_name} CV routing", fontsize=16, pad=14)
        plt.xlabel("Expert ID", fontsize=13)
        plt.ylabel("Layer", fontsize=13)
        plt.tight_layout()

        # plt.show()
        fig.savefig(f"{prefix}/{data_name}/expert_coef_variation.png", dpi=300)
        plt.close()

        # plot of total usage

        fig = plt.figure(figsize=(12, 8))  # larger figure
        sns.heatmap(
            sum_2d,
            cmap="YlGnBu",
            xticklabels=[f"E{i}" for i in range(sum_2d.shape[1])],
            yticklabels=[f"L{i}" for i in range(sum_2d.shape[0])],
            cbar_kws={"label": ""},
        )
        plt.title(f"{data_name} Total Usage", fontsize=16, pad=14)
        plt.xlabel("Expert ID", fontsize=13)
        plt.ylabel("Layer", fontsize=13)
        plt.tight_layout()

        # plt.show()
        fig.savefig(f"{prefix}/{data_name}/expert_usage.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
