import asyncio
import json
import sys
import aiohttp
import random
import argparse
import yaml
import os
import time
from time import sleep

decodesz = 128
# Server URL (replace with your server URL)
SERVER_URL = "http://localhost:10002/v1/chat/completions"
bf_list = [1]
decodesz_list = [128]
prompt_list = ['Please elaborate on modern world history.', 'Please introduce Harry Potter.', 'I want to learn Python. Please give me some advice.', 'Please tell me a joke ']
async def fetch_event_stream(session, payload, request_id):
    try:
        

        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

        async with session.post(SERVER_URL, json=payload, headers=headers, timeout=50000) as response:
            print(f"Request {request_id}: Connected, status {response.status}")

            if response.status != 200:
                print(f"Request {request_id}: Error, status {response.status}")
                return

            output_text = ""  # 存储当前 response 的所有 token
            total_tokens = 0  # 统计总 tokens 数
            decode_start_time = None  # 记录 decode 阶段开始时间
            decode_end_time = None  # 记录 decode 结束时间

            async for line in response.content:
                try:
                    decoded_line = line.decode("utf-8").strip()

                    # 过滤空行
                    if not decoded_line or not decoded_line.startswith("data: "):
                        continue

                    decoded_line = decoded_line[6:].strip()  # 去掉 `data: `

                    # 确保 JSON 数据是合法的
                    if not decoded_line:
                        continue

                    response_data = json.loads(decoded_line)  # 解析 JSON

                    # 确保 choices 存在
                    choices = response_data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")

                    if token:
                        if decode_start_time is None:
                            decode_start_time = time.time()  # 记录 decode 开始时间
                        
                        output_text += token  # 追加 token
                        sys.stdout.write(token)  # 直接输出 token
                        sys.stdout.flush()  # 立即刷新，确保 token 立刻出现在终端
                        total_tokens += 1  # 增加 token 计数
                        decode_end_time = time.time()  # 每次收到 token，更新 decode 结束时间

                    # 检查是否完成
                    finish_reason = choices[0].get("finish_reason", None)
                    if finish_reason:
                        # print(f"\nRequest {request_id}: Done")
                        break  # 结束流式处理

                except json.JSONDecodeError as e:
                    print(f"\nRequest {request_id}: JSON Decode Error - {e}")
                except IndexError:
                    print(f"\nRequest {request_id}: List Index Error - choices is empty")
                except Exception as e:
                    print(f"\nRequest {request_id}: Error parsing stream - {e}")

            # 计算 decode 速度
            if decode_start_time and decode_end_time and total_tokens > 0:
                decode_time = decode_end_time - decode_start_time
                decode_speed = total_tokens / decode_time if decode_time > 0 else 0
                # print(f"Request {request_id}: Decode Speed = {decode_speed:.2f} tokens/s")

    except Exception as e:
        print(f"\nRequest {request_id}: Exception - {e}")

async def main(prompt_id):
    async with aiohttp.ClientSession() as session:
        payload = {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt_list[prompt_id]}
            ],
            "model": "DeepSeek-V3",
            "stream": True,
            "max_completion_tokens": 2,
            # "temperature": 0.3,
            # "top_p": 1.0,     
            # "max_tokens" : 20,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        }
        tasks = [fetch_event_stream(session, payload, prompt_id)]
        await asyncio.gather(*tasks)

        payload["temperature"] = 0.3
        tasks = [fetch_event_stream(session, payload, prompt_id)]
        await asyncio.gather(*tasks)

        payload["top_p"] = 1
        tasks = [fetch_event_stream(session, payload, prompt_id)]
        await asyncio.gather(*tasks)

        payload["max_tokens"] = 200
        tasks = [fetch_event_stream(session, payload, prompt_id)]
        await asyncio.gather(*tasks)
        
        payload["stream"] = False
        tasks = [fetch_event_stream(session, payload, prompt_id)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Event Stream Request Tester")
    parser.add_argument("--question_id", type=int, default=0, required=False)
    args = parser.parse_args()
    output_file = "ktransformer_test_results.txt"
    asyncio.run(main(args.question_id))
