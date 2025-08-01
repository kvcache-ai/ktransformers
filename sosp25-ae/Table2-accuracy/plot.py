import json
import pandas as pd
from collections import defaultdict

def load_data(jsonl_path):
    data = defaultdict(dict)
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            path = entry["load_generations_path"]
            score = round(entry["results"] * 100, 1)

            parts = path.split("/")
            benchmark = parts[-2].upper()
            filename = parts[-1].replace(".jsonl", "")
            model_parts = filename.split("_")
            model_family = model_parts[0].upper()
            layout = model_parts[-1]

            model = model_family.replace("DS3", "DS-3").replace("DS2", "DS-2").replace("QW2", "QW-2")
            model_id = f"{model} ({layout})"

            data[model_id][benchmark] = f"{score:.1f}"
    return data

def make_markdown_table(data, output_path="table2.md"):
    model_order = [
        "DS-3 (8+0)", "DS-3 (2+6)",
        "DS-2 (6+0)", "DS-2 (2+4)",
        "QW-2 (8+0)", "QW-2 (4+4)",
    ]
    benchmarks = ["HUMAN_EVAL", "MBPP", "GSM8K", "STRATEGY_QA"]

    table_data = []
    for model_id in model_order:
        row = []
        for b in benchmarks:
            value = data[model_id].get(b, "--")
            if value != "--":
                value = f"{float(value):.1f}"
            row.append(value)
        table_data.append(row)

    df = pd.DataFrame(table_data, index=model_order, columns=["HumanEval", "MBPP", "GSM8K", "StrategyQA"])
    markdown_table = df.to_markdown(tablefmt="github")
    
    with open(output_path, "w") as f:
        f.write(markdown_table + "\n")
    
def main():
    data = load_data("scores.jsonl")
    make_markdown_table(data)

if __name__ == "__main__":
    main()