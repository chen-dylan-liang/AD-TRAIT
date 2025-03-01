import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Set global font to serif
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ['Times New Roman']
plt.rcParams['font.size'] = 20  # Increase base font size
font = {'fontname': 'Times New Roman'}


def load_data(csv_file, x_col="n"):
    data_dict = defaultdict(lambda: {"x": [], "y": []})

    with open(csv_file, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_indices = {col_name: idx for idx, col_name in enumerate(header)}

        for row in reader:
            if not row:
                continue
            approach = row[col_indices["approach"]]
            try:
                x_val = float(row[col_indices[x_col]])
                time_val = float(row[col_indices["time"]])
                data_dict[f"{approach}_time"]["x"].append(x_val)
                data_dict[f"{approach}_time"]["y"].append(time_val)
            except (ValueError, IndexError):
                continue

    return data_dict


def plot_experiments(jacobian_file, gradient_file):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    approach_mapping_forward = {
        "forward_ad_jax_jit_gpu": "JAX (Python)",
        "forwarddiff": "ForwardDiff.jl (Julia)",
        "enzyme_forward": "Enzyme.jl (Julia)",
        "FAD-AutoDiff-Cpp": "AutoDiff (C++)",
        "FAD-ad trait-Rust": "ad-trait (Rust)",
        "FAD-SIMD-ad trait-Rust": "ad-trait SIMD (Rust)",
    }

    approach_mapping_reverse = {
        "reverse_ad_jax_jit_gpu": "JAX (Python)",
        "reverse_ad_pytorch": "PyTorch (Python)",
        "zygote": "Zygote.jl (Julia)",
        "reversediff": "ReverseDiff.jl (Julia)",
        "enzyme_reverse": "Enzyme.jl (Julia)",
        "RAD-AutoDiff-Cpp": "AutoDiff (C++)",
        "RAD-ad trait-Rust": "ad-trait (Rust)",
        # "RAD-ad trait-Rust": "Burn (Rust)"
    }

    colors = {
        **{k: v for k, v in zip(approach_mapping_forward.keys(), ["#254D3E", "#8B2AB8", "#2F1EC9", "#C96F77", "#0AF211", "#F6FF00"])},
        **{k: v for k, v in zip(approach_mapping_reverse.keys(), ["#254D3E", "#A12020", "#7775D9", "#8B2AB8", "#2F1EC9", "#C96F77", "#0AF211"])}
    }
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    markers = ['o', 's', 'D', '^', 'v', '<', '>']

    forward_approaches = list(approach_mapping_forward.keys())
    reverse_approaches = list(approach_mapping_reverse.keys())

    sub_experiments = [(gradient_file, "Sub-experiment 1"), (jacobian_file, "Sub-experiment 2")]
    for i, (csv_file, title) in enumerate(sub_experiments[0::]):
        try:
            data = load_data(csv_file, "n")
            y_min, y_max = float('inf'), float('-inf')
            for j, (ax, approaches, mapping) in enumerate(zip(axes[i], [forward_approaches, reverse_approaches], [approach_mapping_forward, approach_mapping_reverse])):
                for k, approach in enumerate(approaches):
                    key = f"{approach}_time"
                    if key in data:
                        x_vals = np.array(data[key]["x"])
                        y_vals = np.array(data[key]["y"])
                        sort_idx = np.argsort(x_vals)
                        x_vals = x_vals[sort_idx]
                        y_vals = y_vals[sort_idx]
                        ax.plot(x_vals, y_vals, label=mapping[approach],
                                color=colors[approach], linestyle=line_styles[k % len(line_styles)],
                                marker=markers[k % len(markers)], markersize=6, linewidth=2, alpha=0.9)
                        # y_min = min(y_min, min(y_vals))
                        # y_max = max(y_max, max(y_vals))
                ax.set_yscale("log")
                ax.set_title(f"{title} ({'Forward' if j == 0 else 'Reverse'} Mode)", fontsize=20, pad=10, **font)
                ax.grid(True, linestyle='--', alpha=0.9)
                ax.set_xlabel("n" if i == 0 else "n, m", fontsize=20, **font)
            for ax in axes[0]:
                ax.set_ylim(0.00001, 0.1)
            for ax in axes[1]:
                ax.set_ylim(0.00001, 100)
        except FileNotFoundError:
            print(f"Warning: Could not find {csv_file}")
            for ax in axes[i]:
                ax.text(0.5, 0.5, 'No data available', horizontalalignment='center',
                        verticalalignment='center', fontsize=20)

    axes[0][0].set_ylabel("Average Runtime", fontsize=20, **font)
    axes[1][0].set_ylabel("Average Runtime", fontsize=20, **font)
    # fig.legend(handles=[plt.Line2D([0], [0], color=colors[k], lw=2, marker='o', label=v) for k, v in approach_mapping_forward.items()], fontsize=18, loc='lower left', bbox_to_anchor=(0.025, -0.07), ncol=3)
    # fig.legend(handles=[plt.Line2D([0], [0], color=colors[k], lw=2, marker='o', label=v) for k, v in approach_mapping_reverse.items()], fontsize=18, loc='lower right', bbox_to_anchor=(1.0, -0.105), ncol=3)
    fig.legend(handles=[plt.Line2D([0], [0], color=colors[k], linestyle=line_styles[i % len(line_styles)],
                                   marker=markers[i % len(markers)], lw=2, label=v)
                         for i, (k, v) in enumerate(approach_mapping_forward.items())],
               fontsize=18, loc='lower left', bbox_to_anchor=(0.025, -0.07), ncol=3)
    fig.legend(handles=[plt.Line2D([0], [0], color=colors[k], linestyle=line_styles[i % len(line_styles)],
                                   marker=markers[i % len(markers)], lw=2, label=v)
                         for i, (k, v) in enumerate(approach_mapping_reverse.items())],
               fontsize=18, loc='lower right', bbox_to_anchor=(1.0, -0.105), ncol=3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = plot_experiments("jacobian_results_all.csv", "gradient_results_all.csv")
    plt.savefig('combined_experiment_results.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
