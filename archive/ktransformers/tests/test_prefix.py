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

async def fetch_message_once(session, request_id, messages, max_tokens, model):
    try:
        payload = {
            "messages": messages,
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
                return None, None, None

            buffer = ""
            usage_info = None
            answer = ""

            async for line in response.content:
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
                    buffer += token
                    answer += token

                finish_reason = choices[0].get("finish_reason", None)
                if finish_reason:
                    break

            return answer.strip(), usage_info, buffer.strip()

    except Exception as e:
        print(f"[Request {request_id}] Exception: {e}")
        return None, None, None


async def multi_turn_conversation(session, request_id, rounds, max_tokens, model):
    prompt = ["介绍一下秦始皇", "秦始皇的成就有哪些", "秦始皇的历史影响", "介绍一下秦始皇的陵墓", "秦始皇的统一措施", "秦始皇的政治制度", "秦始皇的文化政策", "秦始皇的军事行动"]
    
    messages = [{"role": "system", "content": ""}]
    global prefill_speeds, decode_speeds

    for i in range(rounds):
        user_msg = f"这是第{i + 1}轮对话，请回答以下问题：{prompt[i % len(prompt)]}"
        messages.append({"role": "user", "content": user_msg})
        print(f"\n[Request {request_id}] >> User: {user_msg}")

        answer, usage_info, _ = await fetch_message_once(session, request_id, messages, max_tokens, model)
        if answer:
            messages.append({"role": "user", "content": answer})
            print(f"[Request {request_id}] << Assistant: {answer}")

        if usage_info:
            prefill_speed = usage_info["prompt_tokens"] / usage_info["prefill_time"]
            decode_speed = usage_info["completion_tokens"] / usage_info["decode_time"]
            prefill_speeds.append(prefill_speed)
            decode_speeds.append(decode_speed)
            print(f'[Request {request_id}] prefill speed: {prefill_speed}')
            print(f'[Request {request_id}] decode speed: {decode_speed}')


async def main(concurrent_requests, rounds, max_tokens, model):
    async with aiohttp.ClientSession() as session:
        tasks = [multi_turn_conversation(session, i, rounds, max_tokens, model) for i in range(concurrent_requests)]
        await asyncio.gather(*tasks)

    if prefill_speeds:
        import numpy as np
        print(f"\n=== Summary ===")
        print(f"Total concurrency: {concurrent_requests}")
        print(f"Avg prefill speed: {np.mean(prefill_speeds)}")
        print(f"Avg decode speed: {np.mean(decode_speeds)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event Stream Request Tester")
    parser.add_argument("--concurrent", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--model", type=str, default="DeepSeek-V3", help="Model name")
    parser.add_argument("--prompt_lens", type=int, default=1024, help="prefill prompt lens, 1024 or 2048")
    parser.add_argument("--api_url", type=str, default="http://localhost:10002/v1/chat/completions", help="API URL")
    parser.add_argument("--max_tokens", type=int, default=50, help="max decode tokens")
    parser.add_argument("--rounds", type=int, default=8, help="Number of multi-turn rounds (before final query)")    
    
    args = parser.parse_args()
    SERVER_URL = args.api_url
    max_tokens = args.max_tokens
    model = args.model

    asyncio.run(main(args.concurrent, args.rounds, max_tokens, model))

