import argparse
import random
import time
import json
import requests
import pandas as pd
from datasets import load_dataset

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['https_proxy'] = ''
os.environ['http_proxy'] = ''
hint = 'There is a single choice question. Answer the question by replying A, B, C, D. No other answers are accepted. Just the letter.'


class DataEvaluator:
    def __init__(self):
        # self.template_prompt = template_prompt
        self.data = []

    def load_data(self, file_path):
        """
        Load data from a Parquet file into a list.
        Each record in the Parquet file should represent an individual record.
        """
        # 读取 Parquet 文件
        # dataset = load_dataset('parquet', data_files=file_path)
        ds = load_dataset(file_path,"all")
        df = pd.DataFrame(ds['test'])
        # print(ds)
        # # ds_1 =  ds['train']
        # ds_2 =  ds['validation']
        # ds_3 =  ds['test']
        # # 将数据集转换为 Pandas DataFrame
        # df_test = pd.DataFrame(ds['test'])
        # df_val = pd.DataFrame(ds['validation'])

        # for _, row in df.iterrows():
        #     self.data.append(row.to_dict())
        # df = pd.read_parquet(file_path)

        for _, row in df.iterrows():
            self.data.append(row.to_dict())

    def get_prompt(self, record):
        """
        Combine fields from a record with the template prompt to create a full prompt.
        :param record: Dictionary containing fields to populate the template.
        :return: A formatted prompt string.
        """
        # 查看ABCD。。。的选项
        options_str = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(record['choices'])])
        prompt = hint + "\nQuestion: " + record['question'] + "\n" + options_str + "\nAnswer: '"
        return prompt
        
    def post_processing(self, text):
        """
        Perform post-processing on the prediction string.
        :param text: The raw prediction string.
        :return: Processed prediction string.
        """
        text = text.lstrip('\n').split('\n')[-1]
        return text[-1:]

    def score(self, pred, answers):
        """
        Calculate scores between the prediction and the answer.
        Uses ROUGE scores as the evaluation metric.
        :param pred: The predicted string.
        :param answer: The reference answer string.
        :return: A dictionary containing ROUGE scores.
        """
        for answer in answers:
            if pred == answer:
                return 1

        return 0

# Function to generate text using API
def generate_text(api_url, question, model_name, stream=False):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        # 添加 API Key
        'Authorization' : 'Bearer '
    }
    data = {
        "messages": [{"content": question, "role": "user"}],
        "model": model_name,
        "stream": stream,
        # "temperature": 0.0
    }
    
    print("POST data:", data)
    response = requests.post(api_url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        return result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
    else:
        print(f"API Request failed with status code {response.status_code}")
        return None

# Main function to handle multiple evaluations
def main(concurrent_requests, data_evaluator: DataEvaluator, result_file, log_file, api_url, model_name):
    start_total_time = time.time()

    total_score = 0

    results = []
   # 设置随机数种子
    random.seed(42)
    random.shuffle(data_evaluator.data)
    for i in range(min(concurrent_requests, len(data_evaluator.data))):
        # Randomly select a data item from data for each request
        data_item = data_evaluator.data[i]
        question = data_evaluator.get_prompt(data_item)
        # print(question)

        # Start the timer for this evaluation
        start_time = time.time()
        try:
            # Generate prediction using the API
            prediction = generate_text(api_url, question, model_name)

            if prediction is None:
                raise Exception(f"Failed to get prediction for {question}")

            answer = chr(data_item['answer'] + 65)
            # Compute score
            score = data_evaluator.score(data_evaluator.post_processing(prediction), answer)

            # Calculate the time taken
            elapsed_time = time.time() - start_time

            # Collect the result data
            result_data = {
                "question_id": i,
                "answer": answer,
                "prediction": data_evaluator.post_processing(prediction),
                "score": score,
                "time": elapsed_time
            }

            # Write results to result.json with each field on a new line
            with open(result_file, 'a', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=4)
                f.write("\n")  # Ensure each JSON object is on a new line

            results.append(result_data)

            # Aggregate scores
            total_score += score

        except Exception as e:
            print(f"Error processing request {i}: {e}")

    # Calculate total time and throughput
    total_time = time.time() - start_total_time
    throughput = concurrent_requests / total_time

    # Log the total time, throughput, and average ROUGE scores
    with open(log_file, 'a', encoding='utf-8') as log_f:
        log_f.write(f"Total Time: {total_time:.2f} seconds\n")
        log_f.write(f"Throughput: {throughput:.2f} requests per second\n")
        log_f.write(f"Average Scores: {total_score / concurrent_requests}\n")
        log_f.write('-' * 40 + '\n')

    print(f"Results saved to {result_file}")
    print(f"Log saved to {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Generate Tester")
    parser.add_argument("--concurrent", type=int, default=1000, help="Number of concurrent evaluations")
    parser.add_argument("--file", type=str, default="cais/mmlu", help="Path to the mmlu.jsonl file")
    parser.add_argument("--result", type=str, default="./mmlu_result_silicon.json", help="Path to save the result JSON file")
    parser.add_argument("--log", type=str, default="./mmlu_result_silicon.log", help="Path to save the log file")
    parser.add_argument("--model", type=str, default="Pro/deepseek-ai/DeepSeek-V3", help="Model name or path")
    parser.add_argument("--api_url", type=str, default="http://localhost:10003/v1/chat/completions", help="API URL")
    # parser.add_argument("--api_url", type=str, default="https://api.siliconflow.cn/v1/chat/completions", help="API URL")

    args = parser.parse_args()

    # Load the data from the provided file
    # template_prompt = hint + "\nQuestion: {question}\nA. {options}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nAnswer: '"
    # template_prompt_pro = hint + "\nQuestion: {question}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nE. {options[4]}\nF. {options[5]}\nG. \
        # {options[6]}\nH. {options[7]}\nI. {options[8]}\nJ. {options[9]}\nAnswer: '"


    # Load the data from the provided file
    data_evaluator = DataEvaluator()
    data_evaluator.load_data(args.file)

    # Run the main function with the specified number of concurrent evaluations
    main(args.concurrent, data_evaluator, args.result, args.log, args.api_url, args.model)