# adapt from https://github.com/abacaj/code-eval?tab=readme-ov-file
import argparse
import json
import os
import time
import requests
import tqdm

from evaluation import filter_answer
from prompts import instruct_prompt
import pandas as pd
from datasets import load_dataset
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def generate_text(api_url,question , model_name, stream=False, auth_token=None):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization' : 'Bearer ' + auth_token if auth_token else ''
    }
    question = instruct_prompt(question)
    data = {
        "messages": [{"content": question, "role": "user"}],
        "model": model_name,
        "stream": stream,
        "temperature": 0.6,
        "max_tokens": 10240,
    }
    print(f"content: {question}")
    response = requests.post(api_url, headers=headers, json=data,verify=False)
    if response.status_code == 200:
        result = response.json()
        results = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        return filter_answer(results)
    else:
        print(f"API Request failed with status code {response.status_code}")
        return None
def load_data(file_path):
        """
        Load data from a Parquet file into a list.
        Each record in the Parquet file should represent an individual record.
        """
        # dataset = load_dataset('parquet', data_files=file_path)
        data = []
        ds = load_dataset(file_path)
        df = pd.DataFrame(ds['train'])
        for _, row in df.iterrows():
            data.append(row.to_dict())
        return data

def get_score(pred, answer):
        """
        Calculate scores between the prediction and the answer.
        Uses ROUGE scores as the evaluation metric.
        :param pred: The predicted string.
        :param answer: The reference answer string.
        :return: A dictionary containing ROUGE scores.
        """
        if pred == answer:
            return 1
        # if we need to compare str with number, convert teh str to number
        try:
            pred = float(pred)
            answer = float(answer)
        except:
            pass
        if pred == answer:
            return 1
        return 0

def run_eval_api(
    api_url: str,
    model_name: str,
    out_path: str,
    format_tabs: bool = False,
    auth_token: str = None,
    problem_file: str = None,
    append: bool = False,
    skip: int = 0
):
  
    data = load_data(problem_file)
    pbar = tqdm.tqdm(total=len(data) * 1)
    pbar.update(skip)
    for i in range(len(data)):
        i = i+skip
        data_item = data[i]
        question = data_item['Problem']
        # Start the timer for this evaluation
        start_time = time.time()
        try:
            completion = generate_text(api_url, question, model_name, auth_token=auth_token)
            if completion is None:
                raise Exception(f"Failed to get prediction for {question}")
            answer = data_item['Answer']
            score = get_score(completion, answer)
            elapsed_time = time.time() - start_time
            result = {
                "index": i,
                "question_id": data_item["ID"],
                "answer": answer,
                "prediction": completion,
                "score": score,
                "time": elapsed_time
            }
            with open(out_path, "a" if append else "w") as f:
                f.write(json.dumps(result) + "\n")
            
        except Exception as e:
            print(f"Failed to get prediction for {question}")
            print(e)
            continue

        pbar.update(1)
    

def main(output_path, api_url, model_name, auth_token, format_tabs,problem_file, append,skip):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    run_eval_api(api_url, model_name, output_path, format_tabs, auth_token, problem_file,append,skip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Generate Tester")
    parser.add_argument("--api_url", type=str, default="https://api.siliconflow.cn/v1/chat/completions", help="API URL")
    parser.add_argument("--model_name", type=str, default="Pro/deepseek-ai/DeepSeek-R1", help="Model Name")
    parser.add_argument("--out_path", type=str, default="results/api/eval_aime.jsonl", help="Output Path")
    parser.add_argument("--auth_token", type=str, default=None, help="Auth Token")
    parser.add_argument("--format_tabs", action="store_true", help="Format Tabs")
    parser.add_argument("--problem_file", type=str, default="Maxwell-Jia/AIME_2024", help="Evalset File")
    parser.add_argument("--no_append", action="store_false", help="Append to existing file")
    parser.add_argument("--skip", type=int, default=0, help="Skip some tasks")
    args = parser.parse_args()
    # api_url = "https://api.siliconflow.cn/v1/chat/completions"
    main(args.out_path, args.api_url, args.model_name, args.auth_token, args.format_tabs, args.problem_file, args.no_append, args.skip)