import json
import matplotlib.pyplot as plt
import numpy as np

models = ["ds3", "ds2", "qw2"]

steps_prefill = ["", "+v", "+m", "+m+d", "+m+d+n", "+m+d+n+c"]
steps_decode = ["", "+v", "+m", "+v+d", "+v+d+n", "+v+d+n+c"]

colors = ['#c7e9c0', '#74c476', '#31a354', '#006d2c', '#00441b', '#003300']

def load_data(filename, key="result", subkey=None):
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            item = json.loads(line)
            config_id = item['config_id']
            if key == "result":
                value = item.get('result', 0.0)
            elif key == "results" and subkey is not None:
                value = item.get('results', {}).get(subkey, 0.0)
            else:
                value = 0.0
            data[config_id] = value
    return data

def prepare_normalized_data(data, steps):
    matrix = []
    for step in steps:
        row = []
        for model in models:
            full_key = model + step if step else model
            base_key = model
            base_val = data.get(base_key, 0)
            curr_val = data.get(full_key, 0)
            if base_val > 0 and curr_val > 0:
                row.append(curr_val / base_val)
            else:
                row.append(0.0)
        matrix.append(row)
    return matrix

def plot_chart(data_matrix, ylabel, filename, legend_labels, steps):
    x = np.arange(len(models))
    bar_width = 0.08
    spacing = 0.03

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, row in enumerate(data_matrix):
        offset = (i - len(data_matrix) / 2) * (bar_width + spacing) + (bar_width + spacing) / 2
        ax.bar(x + offset, row, width=bar_width, color=colors[i], edgecolor='black', label=legend_labels[i])

    ax.set_xticks(x)
    ax.set_xticklabels(['DS-3', 'DS-2', 'QW-2'], fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.grid(axis='y', linestyle='--', color='gray', linewidth=0.7)
    ax.tick_params(axis='y', labelsize=14)

    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.35), fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 1.05])
    plt.savefig(filename + ".pdf")
    plt.show()

def main():
    prefill_data = load_data("prefill_perf.jsonl", key="results", subkey="8192")
    decode_data = load_data("decode_perf.jsonl", key="result")

    prefill_matrix = prepare_normalized_data(prefill_data, steps_prefill)
    decode_matrix = prepare_normalized_data(decode_data, steps_decode)

    prefill_legend = ['Base (Fiddler)', '+v', '+m', '+m+d', '+m+d+n', '+m+d+n+c']
    decode_legend = ['Base (Fiddler)', '+v', '+m', '+v+d', '+v+d+n', '+v+d+n+c']

    plot_chart(prefill_matrix, 'Normalized Prefill Speed', 'figure13_a', prefill_legend, steps_prefill)
    plot_chart(decode_matrix, 'Normalized Decode Speed', 'figure13_b', decode_legend, steps_decode)

if __name__ == "__main__":
    main()