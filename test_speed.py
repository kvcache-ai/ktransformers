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
decodesz_list = [128]
prefill_speeds = []
decode_speeds = []
ktansformer_prompt=""
image_path=""
async def fetch_event_stream(session, request_id, prompt,image_path, max_tokens, model):
    try:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_path
                            },
                        },
                    ],
                }
            ],
            "model": model,
            "temperature": 0.3,
            "top_p": 1.0,
            "stream": True,
            "return_speed": True,
            "max_tokens": max_tokens,
        }

        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

        async with session.post(SERVER_URL, json=payload, headers=headers, timeout=500000) as response:
            if response.status != 200:
                print(f"[Request {request_id}] Error: Status {response.status}")
                return

            buffer = ""  
            total_tokens = 0
            decode_start_time = None
            decode_end_time = None
            usage_info = None  

            async for line in response.content:
                try:
                    decoded_line = line.decode("utf-8").strip()
                    if not decoded_line or not decoded_line.startswith("data: "):
                        continue

                    decoded_line = decoded_line[6:].strip()
                    if not decoded_line:
                        continue

                    response_data = json.loads(decoded_line)
                    
                    if "usage" in response_data:
                        usage_info = response_data["usage"]
                    
                    choices = response_data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")

                    if token:
                        if decode_start_time is None:
                            decode_start_time = time.time()
                        buffer += token
                        total_tokens += 1
                        decode_end_time = time.time()

                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            print(f"[Request {request_id}] {line}")

                    finish_reason = choices[0].get("finish_reason", None)
                    if finish_reason:
                        break

                except Exception as e:
                    print(f"[Request {request_id}] Stream Error: {e}")

            if buffer.strip():
                print(f"[Request {request_id}] {buffer.strip()}")

            if usage_info:
                if "prefill_time" in usage_info:
                    prefill_speed = usage_info["prompt_tokens"] / usage_info["prefill_time"]
                    decode_speed = usage_info["completion_tokens"] / usage_info["decode_time"]
                    prefill_speeds.append(prefill_speed)
                    decode_speeds.append(decode_speed)
                    print(f'[Request {request_id}] prefill speed: {prefill_speed}')
                    print(f'[Request {request_id}] decode speed: {decode_speed}')

    except Exception as e:
        print(f"[Request {request_id}] Exception: {e}")

async def main(concurrent_requests , prompt, image_path,max_tokens, model):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_event_stream(session, i , prompt, image_path, max_tokens, model) for i in range(concurrent_requests)]
        await asyncio.gather(*tasks)
    if len(prefill_speeds) != 0:
        import numpy as np
        print(f"concurrency: {len(prefill_speeds)}")
        print(f"total prefill speed: {np.sum(prefill_speeds)}\n total decode speed: {np.sum(decode_speeds)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event Stream Request Tester")
    parser.add_argument("--concurrent", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--model", type=str, default="DeepSeek-V3", help="Model name")
    parser.add_argument("--prompt_lens", type=int, default=1024, help="prefill prompt lens, 1024 or 2048")
    parser.add_argument("--api_url", type=str, default="http://localhost:10002/v1/chat/completions", help="API URL")
    parser.add_argument("--max_tokens", type=int, default=50, help="max decode tokens")
    
    args = parser.parse_args()
    SERVER_URL = args.api_url
    max_tokens = args.max_tokens
    model = args.model
    prompt = ktansformer_prompt
    asyncio.run(main(args.concurrent, prompt, image_path,max_tokens, model))

