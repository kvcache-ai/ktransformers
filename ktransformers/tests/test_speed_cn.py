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
ktansformer_prompt1024="""在遥远的翡翠森林里，住着各种各样的神奇生物。其中，有一只名叫露露的小狐狸，她与其他狐狸不同，天生长着一双晶莹剔透的翅膀。然而，这双翅膀却从未带她飞翔过。
    一天，森林里传来一个惊人的消息：藏在森林深处的魔法泉水干涸了，所有生物赖以生存的泉水即将枯竭。他们说，只有传说中的“天空之羽”才能唤醒泉水，让它重新流淌。然而，“天空之羽”藏在一座高耸入云的山峰上，没有任何动物能抵达那里。
    露露听到这个消息后，决定亲自去寻找“天空之羽”，即便她的翅膀无法飞翔，她也要尝试。最终，露露来到了传说中的高峰脚下，根本无法攀爬。她望着天空，心里充满了不甘：“如果我能飞起来，就不会被这座山挡住了……”
    正当她感到迷茫时，一只年迈的白鹰出现在她面前。
    “孩子，你为什么到这里来？”白鹰用苍老但慈祥的声音问道。
    露露将森林的困境告诉了白鹰，并说自己愿意付出一切，只要能拯救森林。
    白鹰沉思了一会儿，缓缓说道：“你的翅膀并不是没有力量，而是你一直害怕它们不能飞翔。相信自己，勇敢跳下去。”
    露露听后，心跳加速，她望着万丈深渊，犹豫不决就在那一瞬间，她竟然真的飞了起来！露露兴奋极了，她终于看到了“天空之羽”——一根散发着金光的羽毛，轻盈地悬浮在空中。露露小心翼翼地将“天空之羽”叼住，振翅返回森林。
    当她将羽毛放入干涸的泉水中时，一道金光闪耀。整个森林恢复了生机，花草重新绽放，动物们欢欣鼓舞。从那以后，露露成为了森林的英雄，她是翱翔天空的勇士。她让所有动物都明白：只要相信自己，勇敢前行，就能实现自己的梦想。
    请简述这个故事的内涵 写10000个字。
    在遥远的翡翠森林里，住着各种各样的神奇生物。其中，有一只名叫露露的小狐狸，她与其他狐狸不同，天生长着一双晶莹剔透的翅膀。然而，这双翅膀却从未带她飞翔过。
    一天，森林里传来一个惊人的消息：藏在森林深处的魔法泉水干涸了，所有生物赖以生存的泉水即将枯竭。他们说，只有传说中的“天空之羽”才能唤醒泉水，让它重新流淌。然而，“天空之羽”藏在一座高耸入云的山峰上，没有任何动物能抵达那里。
    露露听到这个消息后，决定亲自去寻找“天空之羽”，即便她的翅膀无法飞翔，她也要尝试。最终，露露来到了传说中的高峰脚下，根本无法攀爬。她望着天空，心里充满了不甘：“如果我能飞起来，就不会被这座山挡住了……”
    正当她感到迷茫时，一只年迈的白鹰出现在她面前。
    “孩子，你为什么到这里来？”白鹰用苍老但慈祥的声音问道。
    露露将森林的困境告诉了白鹰，并说自己愿意付出一切，只要能拯救森林。
    白鹰沉思了一会儿，缓缓说道：“你的翅膀并不是没有力量，而是你一直害怕它们不能飞翔。相信自己，勇敢跳下去。”
    露露听后，心跳加速，她望着万丈深渊，犹豫不决就在那一瞬间，她竟然真的飞了起来！露露兴奋极了，她终于看到了“天空之羽”——一根散发着金光的羽毛，轻盈地悬浮在空中。露露小心翼翼地将“天空之羽”叼住，振翅返回森林。
    当她将羽毛放入干涸的泉水中时，一道金光闪耀。整个森林恢复了生机，花草重新绽放，动物们欢欣鼓舞。从那以后，露露成为了森林的英雄，她是翱翔天空的勇士。她让所有动物都明白：只要相信自己，勇敢前行，就能实现自己的梦想。
    请简述这个故事的内涵 写10000个字。
        露露将森林的困境告诉了白鹰，并说自己愿意付出一切，只要能拯救森林。
    白鹰沉思了一会儿，缓缓说道：“你的翅膀并不是没有力量，而是你一直害怕它们不能飞翔。相信自己，勇敢跳下去。”
    露露听后，心跳加速，她望着万丈深渊，犹豫不决就在那一瞬间，她竟然真的飞了起来！露露兴奋极了，她终于看到了“天空之羽”——一根散发着金光的羽毛，轻盈地悬浮在空中。露露小心翼翼地将“天空之羽”叼住，振翅返回森林。
    当她将羽毛放入干涸的泉水中时，一道金光闪耀。整个森林恢复了生机，花草重新绽放，动物们欢欣鼓舞。从那以后，露露成为了森林的英雄，她是翱翔天空的勇士。她让所有动物都明白：只要相信自己，勇敢前行，就能实现自己的梦想。
    请简述这个故事的内涵 写10000个字。想。
    请简述这个故事的内涵 故事的内涵这个故事的内涵写10000个字"""

prefill_times = []  # 用于存储 prefill 时间
prompts_map = dict()

async def fetch_event_stream(session, request_id , prompt):
    try:
        payload = {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            "model": "DeepSeek-V3",
            "temperature": 0.3,
            "top_p": 1.0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
            "stream": True  # 开启流式输出
        }

        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        request_start_time = time.time()  # ⏱️ 记录请求发出时间

        async with session.post(SERVER_URL, json=payload, headers=headers) as response:
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
                            prefill_time = decode_start_time - request_start_time  # ⏱️ 计算 prefill 时间
                            prefill_times.append(prefill_time)  # ✅ 收集 prefill 时间
                        
                        output_text += token  # 追加 token
                        if request_id not in prompts_map:
                            prompts_map[request_id] = ""
                        prompts_map[request_id] += token
                        sys.stdout.write(str(request_id)) 
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

async def main(concurrent_requests , prompt , prompt_lens):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_event_stream(session, i , prompt) for i in range(concurrent_requests)]
        await asyncio.gather(*tasks)

    for i in range(concurrent_requests):
        if i in prompts_map:
            print(f"\nRequest {i}: Output: {prompts_map[i]}")
        else:
            print(f"\nRequest {i}: No output received.")

    if prefill_times:
        avg_prefill = sum(prefill_times) / len(prefill_times)
        if prompt_lens == 2048:
            print(f"\nAverage prefill time: {avg_prefill:.3f}s, {2349/avg_prefill:.3f} tokens/seconds over {len(prefill_times)} requests.")
        elif prompt_lens == 1024:
            print(f"\nAverage prefill time: {avg_prefill:.3f}s, {1184/avg_prefill:.3f} tokens/seconds over {len(prefill_times)} requests.")
        elif prompt_lens == 10:
            print(f"\nAverage prefill time: {avg_prefill:.3f}s, {26/avg_prefill:.3f} tokens/seconds over {len(prefill_times)} requests.")
    else:
        print("\nNo prefill times recorded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event Stream Request Tester")
    parser.add_argument("--concurrent", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--prompt_lens", type=int, default=1024, help="prefill prompt lens, 1024 or 2048")
    parser.add_argument("--api_url", type=str, default="http://localhost:10002/v1/chat/completions", help="API URL")
    
    args = parser.parse_args()
    SERVER_URL = args.api_url
    if args.prompt_lens == 1024:
        prompt = ktansformer_prompt1024
    elif args.prompt_lens == 2048:
        prompt = ktansformer_prompt1024 * 2
    else:
        prompt = ktansformer_prompt1024[:args.prompt_lens]
    asyncio.run(main(args.concurrent, prompt, args.prompt_lens))

