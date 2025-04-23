import argparse
import random
import time
import json
import requests
import pandas as pd
from datasets import load_dataset
import os
import concurrent.futures
import threading
import re

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['https_proxy'] = ''
os.environ['http_proxy'] = ''
hint = 'There is a single choice question. Answer the question by replying A, B, C, D. No other answers are accepted. Just the letter.'


def extract_final_answer(text):
    """
    提取模型预测的最终选项（如 A/B/C/D）
    支持自然语言、多行、markdown、高亮、非末尾结论等格式
    """
    text = text.strip()

    # 1. 显式语句匹配（优先）
    explicit_patterns = [
        r'Answer:\s*([A-D])\b',
        r'Correct answer:\s*([A-D])\b',
        r'The correct answer is\s*\*?\*?\s*([A-D])\b',
        r'Answer is\s*([A-D])\b',
        r'Therefore,\s*answer is\s*([A-D])\b',
        r'Therefore,\s*the answer should be\s*(?:Option\s*)?([A-D])\b',
        r'The answer should be\s*(?:Option\s*)?([A-D])\b',
        r'Option\s+([A-D])\s+is correct',
    ]
    for pat in explicit_patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # 2. markdown 强调 **C**, **C. something**
    markdown_match = re.findall(r'\*\*\s*([A-D])[\.\s]?', text)
    if markdown_match:
        return markdown_match[-1].upper()

    # 3. 查找单引号中的 'C' 或 "C"
    quote_match = re.findall(r"['\"]([A-D])['\"]", text)
    if quote_match:
        return quote_match[-1].upper()

    # 4. 倒数几行是否以 "C." 或 "C" 开头
    lines = text.splitlines()
    for line in reversed(lines[-5:]):
        line = line.strip()
        match = re.match(r'^([A-D])([.\s]|$)', line)
        if match:
            return match.group(1).upper()
    
    # 再不行就返回 None
    return None
class DataEvaluator:
    def __init__(self):
        self.data = []

    def load_data(self, file_path):
        """
        从数据文件中加载数据，每条记录对应一个实例
        """
        splits = {'test': 'all/test-00000-of-00001.parquet', 'validation': 'all/validation-00000-of-00001.parquet',
                  'dev': 'all/dev-00000-of-00001.parquet',
                  'auxiliary_train': 'all/auxiliary_train-00000-of-00001.parquet'}
        df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])
        for _, row in df.iterrows():
            self.data.append(row.to_dict())

    def get_prompt(self, record):
        """
        结合提示信息和记录数据生成完整的题目
        """
        options_str = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(record['choices'])])
        prompt = hint + "\nQuestion: " + record['question'] + "\n" + options_str + "\nAnswer: '"
        return prompt

    def post_processing(self, text):
        """
        对生成的文本进行后处理，提取最终答案（只返回最后一个字符）
        """
        text = text.lstrip('\n').split('\n')[-1]
        return text[-1:]

    def score(self, pred, answer):
        """
        对比预测答案和正确答案，返回得分
        """
        if pred == answer:
            return 1
        return 0

def generate_text(api_url, question, model_name, stream=False):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'Bearer '  # 如有需要，请填入 API Key
    }
    data = {
        "messages": [{"content": question, "role": "user"}],
        "model": model_name,
        "stream": stream,
    }
    print("POST data:", data)
    response = requests.post(api_url, headers=headers, json=data, timeout=5000000)
    if response.status_code == 200:
        result = response.json()
        return result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
    else:
        print(f"API Request failed with status code {response.status_code}")
        return None

def main(concurrent_requests, data_evaluator: DataEvaluator, result_file, log_file, api_url, model_name):
    start_total_time = time.time()
    total_score = 0
    total_exact_score = 0
    results = []
    file_lock = threading.Lock()
    
    # 打乱数据顺序，并选择需要测试的实例数
    random.seed(42)
    random.shuffle(data_evaluator.data)
    data_subset = data_evaluator.data[:min(concurrent_requests, len(data_evaluator.data))]
    
    batch_size = 10  # 每批次最多 10 个实例

    def worker(index, data_item):
        nonlocal total_score
        nonlocal total_exact_score
        question = data_evaluator.get_prompt(data_item)
        start_time = time.time()
        try:
            prediction = generate_text(api_url, question, model_name)
            if prediction is None:
                raise Exception(f"Failed to get prediction for question: {question}")
            # 正确答案：将数字转换成字母（0->A, 1->B, 2->C, 3->D）
            answer = chr(data_item['answer'] + 65)
            processed_prediction = data_evaluator.post_processing(prediction)
            score = data_evaluator.score(processed_prediction, answer)
            exact_score = data_evaluator.score(extract_final_answer(prediction), answer)
            elapsed_time = time.time() - start_time
            result_data = {
                "question_id": index,
                "answer": answer,
                "prediction": processed_prediction,
                "full_prediction": prediction,
                "score": score,
                "exact_score": exact_score,
                "time": elapsed_time
            }
            # 写入结果时加锁保证线程安全
            with file_lock:
                with open(result_file, 'a', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=4)
                    f.write("\n")
            return result_data
        except Exception as e:
            print(f"Error processing request {index}: {e}")
            return None

    # 按批次处理，每批最多 10 个任务
    for batch_start in range(0, len(data_subset), batch_size):
        batch = data_subset[batch_start: batch_start + batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(worker, batch_start + j, data_item) for j, data_item in enumerate(batch)]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res is not None:
                    results.append(res)
                    total_score += res['score']
                    total_exact_score += res['exact_score']
    
    total_time = time.time() - start_total_time
    throughput = len(data_subset) / total_time if total_time > 0 else 0
    
    with open(log_file, 'a', encoding='utf-8') as log_f:
        log_f.write(f"Total Time: {total_time:.2f} seconds\n")
        log_f.write(f"Throughput: {throughput:.2f} requests per second\n")
        average_score = total_score / len(data_subset) if data_subset else 0
        log_f.write(f"Average Score: {average_score}\n")
        average_exact_score = total_exact_score / len(data_subset) if data_subset else 0
        log_f.write(f"Average Exact Score: {average_exact_score}\n")
        log_f.write('-' * 40 + '\n')
    
    print(f"Results saved to {result_file}")
    print(f"Log saved to {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Generate Tester")
    parser.add_argument("--concurrent", type=int, default=1000, help="需要测试的实例总数")
    parser.add_argument("--file", type=str, default="cais/mmlu", help="数据文件路径")
    parser.add_argument("--result", type=str, default="./mmlu_result_silicon.json", help="结果文件保存路径")
    parser.add_argument("--log", type=str, default="./mmlu_result_silicon.log", help="日志文件保存路径")
    parser.add_argument("--model", type=str, default="Pro/deepseek-ai/DeepSeek-V3", help="模型名称或路径")
    parser.add_argument("--api_url", type=str, default="http://localhost:10006/v1/chat/completions", help="API URL")

    args = parser.parse_args()
    
    data_evaluator = DataEvaluator()
    data_evaluator.load_data(args.file)
    
    main(args.concurrent, data_evaluator, args.result, args.log, args.api_url, args.model)