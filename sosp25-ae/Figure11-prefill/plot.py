import matplotlib.pyplot as plt
import numpy as np
import json

def parse_data_from_json(json_data):
    model_order = [
        'DS-3 BF16/FP16', 'DS-2 BF16/FP16', 'QW-2 BF16/FP16',
        'DS-3 Int4', 'DS-2 Int8', 'QW-2 Int8'
    ]
    
    model_to_configs = {
        'DS-3 BF16/FP16': ['ktransformers_ds3_bf16', 'fiddler_ds3_bf16', 'llama.cpp_ds3_fp16'],
        'DS-3 Int4': ['ktransformers_ds3_int4', 'llama.cpp_ds3_int4'],
        'DS-2 BF16/FP16': ['ktransformers_ds2_bf16', 'fiddler_ds2_bf16', 'llama.cpp_ds2_fp16'],
        'DS-2 Int8': ['ktransformers_ds2_int8', 'llama.cpp_ds2_int8'],
        'QW-2 BF16/FP16': ['ktransformers_qw2_bf16', 'fiddler_qw2_bf16', 'llama.cpp_qw2_fp16'],
        'QW-2 Int8': ['ktransformers_qw2_int8', 'llama.cpp_qw2_int8'],
    }
    
    config_to_framework = {
        'ktransformers_ds3_bf16': 'KTransformers',
        'ktransformers_ds3_int4': 'KTransformers',
        'ktransformers_ds2_bf16': 'KTransformers',
        'ktransformers_ds2_int8': 'KTransformers',
        'ktransformers_qw2_bf16': 'KTransformers',
        'ktransformers_qw2_int8': 'KTransformers',
        'fiddler_ds3_bf16': 'Fiddler',
        'fiddler_ds2_bf16': 'Fiddler',
        'fiddler_qw2_bf16': 'Fiddler',
        'llama.cpp_ds3_fp16': 'Llama.cpp',
        'llama.cpp_ds3_int4': 'Llama.cpp',
        'llama.cpp_ds2_fp16': 'Llama.cpp',
        'llama.cpp_ds2_int8': 'Llama.cpp',
        'llama.cpp_qw2_fp16': 'Llama.cpp',
        'llama.cpp_qw2_int8': 'Llama.cpp',
    }
    
    config_data_dict = {}
    for config_data in json_data:
        config_id = config_data['config_id']
        config_data_dict[config_id] = config_data['results']
    
    data = {}
    x_labels = ['32', '64', '128', '256', '512', '1024', '2048', '4096', '8192']
    zero_data = [0.0] * len(x_labels)
    
    for model_name in model_order:
        data[model_name] = {
            'Fiddler': zero_data.copy(),
            'Llama.cpp': zero_data.copy(),
            'KTransformers': zero_data.copy()
        }
        
        for config_id in model_to_configs[model_name]:
            if config_id in config_data_dict:
                framework = config_to_framework[config_id]
                results = config_data_dict[config_id]
                
                framework_data = []
                for token_length in x_labels:
                    value = results.get(token_length, 0)
                    framework_data.append(float(value))
                
                data[model_name][framework] = framework_data
    
    return data, x_labels, model_order

def load_data_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    json_lines = content.split('\n')
    json_data = []
    for line in json_lines:
        if line.strip():
            json_data.append(json.loads(line))
    
    return json_data

def plot_data(data, x_labels, model_list, save_prefix="figure11"):
    bar_width = 0.25
    colors = ['lightgray', 'sandybrown', 'mediumseagreen']
    hatches = ['////', '...', 'xxx']
    frameworks = ['Fiddler', 'Llama.cpp', 'KTransformers']
    
    x = np.arange(len(x_labels))
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 5.6))
    axs = axs.flatten()
    
    for idx, model_name in enumerate(model_list):
        ax = axs[idx]
        model_data = data[model_name]
        used_frameworks = [f for f in frameworks if f in model_data]
        num_bars = len(used_frameworks)
        
        for i, framework in enumerate(used_frameworks):
            offset = x + (i - (num_bars - 1)/2) * bar_width
            color = colors[frameworks.index(framework)]
            hatch = hatches[frameworks.index(framework)]
            ax.bar(offset, model_data[framework], bar_width,
                   label=framework if idx == 0 else "",
                   color=color, edgecolor='black', hatch=hatch)
        
        ax.set_title(model_name, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8.5)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.5)
        
        if idx in [3, 4, 5]:
            ax.set_xlabel('Prompt Length (Tokens)', fontsize=12)
        if idx in [0, 3]:
            ax.set_ylabel('Prefill Speed (Tokens/s)', fontsize=12)
    
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=12, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{save_prefix}.pdf", dpi=300)
    plt.show()

def main():
    json_data = load_data_from_file('prefill_perf.jsonl')
    data, x_labels, model_list = parse_data_from_json(json_data)
    plot_data(data, x_labels, model_list)
        

if __name__ == "__main__":
    main()