import asyncio
import json
import sys
import aiohttp
import argparse

prompt_list = [
    'Please elaborate on modern world history.',
    'Please introduce Harry Potter.',
    'I want to learn Python. Please give me some advice.',
    'Please tell me a joke '
]


async def fetch_event_stream(session, payload, request_id, stream):
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

            output_text = ""

            if stream:
                async for line in response.content:
                    try:
                        decoded_line = line.decode("utf-8").strip()
                        if not decoded_line or not decoded_line.startswith("data: "):
                            continue

                        decoded_line = decoded_line[6:].strip()
                        if not decoded_line:
                            continue

                        response_data = json.loads(decoded_line)
                        choices = response_data.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        token = delta.get("content", "")

                        if token:
                            output_text += token
                            sys.stdout.write(token)
                            sys.stdout.flush()

                        finish_reason = choices[0].get("finish_reason", None)
                        if finish_reason:
                            break

                    except json.JSONDecodeError as e:
                        print(f"\nRequest {request_id}: JSON Decode Error - {e}")
                    except IndexError:
                        print(f"\nRequest {request_id}: List Index Error - choices is empty")
                    except Exception as e:
                        print(f"\nRequest {request_id}: Error parsing stream - {e}")
            else:
                # 非 stream 模式下，一次性接收完整 json
                response_data = await response.json()
                choices = response_data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    print(f"Request {request_id} Output:\n{content}")
                    output_text += content

    except Exception as e:
        print(f"\nRequest {request_id}: Exception - {e}")

async def main(prompt_id, model, stream, max_tokens, temperature, top_p):
    async with aiohttp.ClientSession() as session:
        payload = {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt_list[prompt_id]}
            ],
            "model": model,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        tasks = [fetch_event_stream(session, payload, prompt_id, stream)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event Stream Request Tester")
    parser.add_argument("--question_id", type=int, default=0)
    parser.add_argument("--model", type=str, default="DeepSeek-V3")
    parser.add_argument("--stream", type=bool, default=True)  
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--api_url", type=str, default="http://localhost:10002/v1/chat/completions", help="API URL")

    args = parser.parse_args()
    SERVER_URL = args.api_url
    asyncio.run(main(args.question_id, args.model, args.stream, args.max_tokens, args.temperature, args.top_p))
