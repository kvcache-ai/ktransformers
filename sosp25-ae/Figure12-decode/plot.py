import matplotlib.pyplot as plt
import numpy as np
import json

def parse_data_from_json(json_data):
    bf16_models = ["DS-3", "DS-2", "QW-2"]
    quantized_models = ["DS-3 Int4", "DS-2 Int8", "QW-2 Int8"]
    
    model_to_configs = {
        'DS-3': ['ktransformers_ds3_bf16_8+0', 'ktransformers_ds3_bf16_5+3', 'fiddler_ds3_bf16', 'llama.cpp_ds3_fp16'],
        'DS-2': ['ktransformers_ds2_bf16_6+0', 'ktransformers_ds2_bf16_2+4', 'fiddler_ds2_bf16', 'llama.cpp_ds2_fp16'],
        'QW-2': ['ktransformers_qw2_bf16_8+0', 'ktransformers_qw2_bf16_6+2', 'fiddler_qw2_bf16', 'llama.cpp_qw2_fp16'],
        'DS-3 Int4': ['ktransformers_ds3_int4_8+0', 'ktransformers_ds3_int4_2+6', 'llama.cpp_ds3_int4'],
        'DS-2 Int8': ['ktransformers_ds2_int8_6+0', 'ktransformers_ds2_int8_2+4', 'llama.cpp_ds2_int8'],
        'QW-2 Int8': ['ktransformers_qw2_int8_8+0', 'ktransformers_qw2_int8_4+4', 'llama.cpp_qw2_int8'],
    }
    
    config_to_framework = {
        'ktransformers_ds3_bf16_8+0': 'KTransformers',
        'ktransformers_ds3_int4_8+0': 'KTransformers',
        'ktransformers_ds2_bf16_6+0': 'KTransformers',
        'ktransformers_ds2_int8_6+0': 'KTransformers',
        'ktransformers_qw2_bf16_8+0': 'KTransformers',
        'ktransformers_qw2_int8_8+0': 'KTransformers',
        'ktransformers_ds3_bf16_5+3': 'KTransformers + Expert Deferral',
        'ktransformers_ds3_int4_2+6': 'KTransformers + Expert Deferral',
        'ktransformers_ds2_bf16_2+4': 'KTransformers + Expert Deferral',
        'ktransformers_ds2_int8_2+4': 'KTransformers + Expert Deferral',
        'ktransformers_qw2_bf16_6+2': 'KTransformers + Expert Deferral',
        'ktransformers_qw2_int8_4+4': 'KTransformers + Expert Deferral',
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
        if 'decode_speed' in config_data:
            config_data_dict[config_id] = config_data['decode_speed']
        elif 'result' in config_data and isinstance(config_data['result'], (int, float)):
            config_data_dict[config_id] = float(config_data['result'])
        elif 'result' in config_data and isinstance(config_data['result'], dict):
            values = list(config_data['result'].values())
            if values:
                config_data_dict[config_id] = float(values[0])
    
    bf16_data = {
        'Fiddler': [],
        'Llama.cpp': [],
        'KTransformers': [],
        'KTransformers + Expert Deferral': [],
    }
    
    int_data = {
        'Llama.cpp': [],
        'KTransformers': [],
        'KTransformers + Expert Deferral': [],
    }
    
    for model_name in bf16_models:
        model_values = {
            'Fiddler': 0.0,
            'Llama.cpp': 0.0,
            'KTransformers': 0.0,
            'KTransformers + Expert Deferral': 0.0,
        }
        
        for config_id in model_to_configs[model_name]:
            if config_id in config_data_dict:
                framework = config_to_framework[config_id]
                model_values[framework] = config_data_dict[config_id]
        
        for framework in bf16_data.keys():
            bf16_data[framework].append(model_values[framework])
    
    for model_name in quantized_models:
        model_values = {
            'Llama.cpp': 0.0,
            'KTransformers': 0.0,
            'KTransformers + Expert Deferral': 0.0,
        }
        
        for config_id in model_to_configs[model_name]:
            if config_id in config_data_dict:
                framework = config_to_framework[config_id]
                model_values[framework] = config_data_dict[config_id]
        
        for framework in int_data.keys():
            int_data[framework].append(model_values[framework])
    
    return bf16_data, int_data, bf16_models, quantized_models

def load_data_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    json_lines = content.split('\n')
    json_data = []
    for line in json_lines:
        if line.strip():
            json_data.append(json.loads(line))
    
    return json_data

def plot_data(bf16_data, int_data, bf16_models, quantized_models, save_prefix="figure12"):
    color_map = {
        "Fiddler": 'lightgray',
        "Llama.cpp": 'sandybrown',
        "KTransformers": 'mediumseagreen',
        "KTransformers + Expert Deferral": 'CornflowerBlue',
    }
    hatch_map = {
        "Fiddler": '////',
        "Llama.cpp": '...',
        "KTransformers": 'xxx',
        "KTransformers + Expert Deferral": '\\\\\\\\',
    }
    
    x = np.arange(len(bf16_models))
    
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))
    titles = ["BF16/FP16 Models", "Quantized Models"]
    datasets = [bf16_data, int_data]
    model_labels = [bf16_models, quantized_models]
    
    for ax, title, data_dict, labels in zip(axs, titles, datasets, model_labels):
        keys = list(data_dict.keys())
        bar_width = 0.15
        
        for i, key in enumerate(keys):
            offset = x + (i - (len(keys) - 1) / 2) * bar_width
            ax.bar(offset, data_dict[key], bar_width,
                   label=key, color=color_map[key],
                   edgecolor='black', hatch=hatch_map[key])
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('Decode Speed (Tokens/s)', fontsize=13)
        ax.set_title(title, fontsize=13)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.5)
        ax.tick_params(axis='y', labelsize=12)
    
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.84])
    plt.savefig(f"{save_prefix}.pdf", dpi=300)
    plt.show()

def main():
    json_data = load_data_from_file('decode_perf.jsonl')
    bf16_data, int_data, bf16_models, quantized_models = parse_data_from_json(json_data)
    plot_data(bf16_data, int_data, bf16_models, quantized_models)

if __name__ == "__main__":
    main()