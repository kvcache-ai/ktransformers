# adapt from https://github.com/abacaj/code-eval?tab=readme-ov-file
import argparse
import os
import requests
from human_eval.data import write_jsonl, read_problems
import tqdm

from evaluation import filter_code, fix_indents
from prompts import instruct_prompt

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
        "temperature": 0.6
    }
    print(f"content: {question}")
    response = requests.post(api_url, headers=headers, json=data,verify=False)
    if response.status_code == 200:
        result = response.json()
        results = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        return [filter_code(fix_indents(results))]
    else:
        print(f"API Request failed with status code {response.status_code}")
        return None

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
    if(problem_file is None):
        problems = read_problems()
    else:
        problems = read_problems(problem_file)
    samples = []
    pbar = tqdm.tqdm(total=len(problems) * 1)
    pbar.update(skip)
    try:
        for task_id in problems:
            # skip some tasks
            if skip > 0:
                skip -= 1
                continue

            if format_tabs:
                prompt = problems[task_id]["prompt"].replace("    ", "\t")
            else:
                prompt = problems[task_id]["prompt"]
            completion = generate_text(api_url, prompt, model_name, auth_token=auth_token)
            # samples.append({"task_id": task_id, "completion": completion})
            for sample in completion:
                result = dict(
                    task_id=task_id,
                    completion=sample,
                )
                samples += [result]
                if append:
                    write_jsonl(out_path, [result],append=append)
            pbar.update(1)
        if not append:
            write_jsonl(out_path, samples,append=append)
    except Exception as e:
        if not append:
            write_jsonl(out_path, samples,append=append)
        print(f"Error: {e}")

def main(output_path, api_url, model_name, auth_token, format_tabs,problem_file, append,skip):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    run_eval_api(api_url, model_name, output_path, format_tabs, auth_token, problem_file,append,skip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Generate Tester")
    #parser.add_argument("--api_url", type=str, default="https://api.siliconflow.cn/v1/chat/completions", help="API URL")
    parser.add_argument("--api_url", type=str, default="http://localhost:10002/v1/chat/completions", help="API URL")
    parser.add_argument("--model_name", type=str, default="Pro/deepseek-ai/DeepSeek-V3", help="Model Name")
    parser.add_argument("--out_path", type=str, default="results/api/eval_b.jsonl", help="Output Path")
    parser.add_argument("--auth_token", type=str, default=None, help="Auth Token")
    parser.add_argument("--format_tabs", action="store_true", help="Format Tabs")
    parser.add_argument("--problem_file", type=str, default=None, help="Evalset File")
    parser.add_argument("--no_append", action="store_false", help="Append to existing file")
    parser.add_argument("--skip", type=int, default=0, help="Skip first n problems")
    args = parser.parse_args()
    # api_url = "https://api.siliconflow.cn/v1/chat/completions"
    main(args.out_path, args.api_url, args.model_name, args.auth_token, args.format_tabs, args.problem_file, args.no_append,args.skip)