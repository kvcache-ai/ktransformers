import json

converted_data = []
with open('/data/user23202791/lpl/LLaMA-Factory/examples/KT_used/translation.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        converted_data.append({
            "instruction": "",
            "input": data["问"],
            "output": data["答"]
        })

with open('/data/user23202791/lpl/LLaMA-Factory/examples/KT_used/sft_translation.json', 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)