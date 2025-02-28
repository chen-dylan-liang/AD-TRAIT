import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Set global font to serif
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ['Times New Roman']
plt.rcParams['font.size'] = 12  # Increase base font size
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


def plot_experiments(subexp1_file):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 8))

    approach_mapping = {
        "enzyme_reverse": "RAD-Enzyme-Julia",
        "enzyme_forward": "FAD-Enzyme-Julia",
        "zygote": "RAD-Zygote-Julia",
        "reversediff": "RAD-ReverseDiff-Julia",
        "forwarddiff": "FAD-ForwardDiff-Julia",
    }

    approaches = list(approach_mapping.keys())
    colors = {
        "enzyme_reverse": "#874F8D",
        "enzyme_forward": "#1C6AB1",
        "zygote": "#A86A9D",
        "reversediff": "#0D95CE",
        "forwarddiff": "#ED4043",
    }

    line_styles = ['--', '--', '-', '-', '-.']
    markers = ['o', 's', 'D', '^', 'v']
    approach_styles = {approach: (colors[approach], ls, marker)
                       for approach, ls, marker in zip(approaches, line_styles, markers)}

    try:
        data = load_data(subexp1_file, "n")
        for approach in approaches:
            key = f"{approach}_time"
            if key in data:
                x_vals = np.array(data[key]["x"])
                y_vals = np.array(data[key]["y"])

                sort_idx = np.argsort(x_vals)
                x_vals = x_vals[sort_idx]
                y_vals = y_vals[sort_idx]

                color, ls, marker = approach_styles[approach]
                ax.plot(x_vals, y_vals, label=approach_mapping[approach],
                        color=color, linestyle=ls, marker=marker,
                        markersize=8, linewidth=2, alpha=0.6)

        ax.set_yscale("log")
        ax.set_xlabel("n", fontsize=14, **font)
        ax.set_ylabel("Average Runtime", fontsize=14, **font)
        ax.set_title("Runtime Comparison", fontsize=16, pad=10, **font)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)

    except FileNotFoundError:
        print(f"Warning: Could not find {subexp1_file}")
        ax.text(0.5, 0.5, 'No data available', horizontalalignment='center',
                verticalalignment='center', fontsize=14)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = plot_experiments("Julia/autodiff_benchmark_results.csv")
    plt.savefig('comparison_plot_fixed.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
