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
ktansformer_prompt1024="""Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. 
They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense.Mr. Dursley was the director of a firm called Grunnings, which made drills. 
He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. 
Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors. 
The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere.
The Dursleys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. 
They didn't think they could bear it if anyone found out about the Potters. Mrs. Potter was Mrs. Dursley's sister, but they hadn't met for several years; in fact, Mrs. 
Dursley pretended she didn't have a sister, because her sister and her good-for-nothing husband were as unDursleyish as it was possible to be. 
The Dursleys shuddered to think what the neighbors would say if the Potters arrived in the street. 
The Dursleys knew that the Potters had a small son, too, but they had never even seen him. 
This boy was another good reason for keeping the Potters away; they didn't want Dudley mixing with a child like that.When Mr. and Mrs. 
Dursley woke up on the dull, gray Tuesday our story starts, there was nothing about the cloudy sky outside to suggest that strange and mysterious things would soon be happening all over the country. 
Mr. Dursley hummed as he picked out his most boring tie for work, and Mrs. Dursley gossiped away happily as she wrestled a screaming Dudley into his high chair.None of them noticed a large, tawny owl flutter past the window.
At half past eight, Mr. Dursley picked up his briefcase, pecked Mrs. Dursley on the cheek, and tried to kiss Dudley good-bye but missed, because Dudley was now having a tantrum and throwing his cereal at the walls.
“Little tyke,” chortled Mr. Dursley as he left the house. He got into his car and backed out of number four's drive.
It was on the corner of the street that he noticed the first sign of something peculiar — a cat reading a map. 
For a second, Mr. Dursley didn't realize what he had seen — then he jerked his head around to look again. 
There was a tabby cat standing on the corner of Privet Drive, but there wasn't a map in sight. 
What could he have been thinking of? It must have been a trick of the light. 
Mr. Dursley blinked and stared at the cat. It stared back. As Mr. Dursley drove around the corner and up the road, he watched the cat in his mirror. 
It was now reading the sign that said Privet Drive — no, looking at the sign; cats couldn't read maps or signs. 
Mr. Dursley gave himself a little shake and put the cat out of his mind. 
As he drove toward town he thought of nothing except a large order of drills he was hoping to get that day.
But on the edge of town, drills were driven out of his mind by something else. 
As he sat in the usual morning traffic jam, he couldn't help noticing that there seemed to be a lot of strangely dressed people about. 
People in cloaks. Mr. Dursley couldn't bear people who dressed in funny clothes — the getups you saw on young people! 
He supposed this was some stupid new fashion. He drummed his fingers on the steering wheel and his eyes fell on a huddle of these weirdos standing quite close by. 
They were whispering excitedly together. Mr. Dursley was enraged to see that a couple of them weren't young at all; why, that man had to be older than he was, and wearing an emerald-green cloak! 
The nerve of him! But then it struck Mr. Dursley that this was probably some silly stunt — these people were obviously collecting for something… yes, that would be it. 
The traffic moved on and a few minutes later, Mr. Dursley arrived in the Grunnings parking lot, his mind back on drills.
Mr. Dursley always sat with his back to the window in his office on the ninth floor."""
async def fetch_event_stream(session, request_id, prompt, max_tokens, model):
    try:
        payload = {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
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
                    # print(f"[Request {request_id}] Usage:")
                    # for key, value in usage_info.items():
                    #     print(f"  {key}: {value}")
                    prefill_speed = usage_info["prompt_tokens"] / usage_info["prefill_time"]
                    decode_speed = usage_info["completion_tokens"] / usage_info["decode_time"]
                    prefill_speeds.append(prefill_speed)
                    decode_speeds.append(decode_speed)
                    print(f'[Request {request_id}] prefill speed: {prefill_speed}')
                    print(f'[Request {request_id}] decode speed: {decode_speed}')

    except Exception as e:
        print(f"[Request {request_id}] Exception: {e}")

async def main(concurrent_requests , prompt, max_tokens, model):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_event_stream(session, i , prompt, max_tokens, model) for i in range(concurrent_requests)]
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
    if args.prompt_lens == 1024:
        prompt = ktansformer_prompt1024
    elif args.prompt_lens == 2048:
        prompt = ktansformer_prompt1024 * 2
    elif args.prompt_lens == 4096:
        prompt = ktansformer_prompt1024 * 4
    asyncio.run(main(args.concurrent, prompt, max_tokens, model))

