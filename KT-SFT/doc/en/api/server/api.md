# API

- [OpenAI ChatCompletion](#openai-chatcompletion)
- [Ollama ChatCompletion](#ollama-chatcompletion)
- [OpenAI Assistant](#openai-assistant)

## OpenAI ChatCompletion
```bash
POST /v1/chat/completions

```
Generate responses based on the selected model.

### Parameters
- `messages`: An array of `message` representing all historical messages. A `message` can be from a user or model (assistant) and includes:

  - `role`: Either `user` or `assistant`, indicating the creator of this message.
  - `content`: The message from the user or model.
- `model`: The name of the selected model
- `stream`: Either true or false. Indicates whether to use streaming response. If true, model inference results are returned via HTTP event stream.

### Response
- Streaming response: An event stream, each event contains a `chat.completion.chunk`. `chunk.choices[0].delta.content` is the incremental output returned by the model each time.
- Non-streaming response: Not supported yet.



### Example

```bash
curl -X 'POST' \
  'http://localhost:9112/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {
      "content": "tell a joke",
      "role": "user"
    }
  ],
  "model": "Meta-Llama-3-8B-Instruct",
  "stream": true
}'
```

```bash
data:{"id":"c30445e8-1061-4149-a101-39b8222e79e1","object":"chat.completion.chunk","created":1720511671,"model":"not implmented","system_fingerprint":"not implmented","usage":null,"choices":[{"index":0,"delta":{"content":"Why ","role":"assistant","name":null},"logprobs":null,"finish_reason":null}]}

data:{"id":"c30445e8-1061-4149-a101-39b8222e79e1","object":"chat.completion.chunk","created":1720511671,"model":"not implmented","system_fingerprint":"not implmented","usage":null,"choices":[{"index":0,"delta":{"content":"","role":"assistant","name":null},"logprobs":null,"finish_reason":null}]}

data:{"id":"c30445e8-1061-4149-a101-39b8222e79e1","object":"chat.completion.chunk","created":1720511671,"model":"not implmented","system_fingerprint":"not implmented","usage":null,"choices":[{"index":0,"delta":{"content":"couldn't ","role":"assistant","name":null},"logprobs":null,"finish_reason":null}]}

...

data:{"id":"c30445e8-1061-4149-a101-39b8222e79e1","object":"chat.completion.chunk","created":1720511671,"model":"not implmented","system_fingerprint":"not implmented","usage":null,"choices":[{"index":0,"delta":{"content":"two-tired!","role":"assistant","name":null},"logprobs":null,"finish_reason":null}]}

event: done
data: [DONE]
```



## Ollama ChatCompletion

```bash
POST /api/generate
```

Generate responses using the selected model.

### Parameters
- `prompt`: A string representing the input prompt.
- `model`: The name of the selected model
- `stream`: Either true or false. Indicates whether to use streaming responses. If true, returns the model inference results in the form of an HTTP event stream.

### Response
- Streaming response: A stream of JSON responses, each line is a JSON.
  - `response`: The incremental result of the model completion.
  - `done`: Whether the inference has finished.
- Non-streaming response: Not yet supported.

### 例子

```bash
curl -X 'POST' \
  'http://localhost:9112/api/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "Meta-Llama-3-8B-Instruct",
  "prompt": "tell me a joke",
  "stream": true
}'
```

```bash
{"model":"Meta-Llama-3-8B-Instruct","created_at":"2024-07-09 08:13:11.686513","response":"I'll ","done":false}
{"model":"Meta-Llama-3-8B-Instruct","created_at":"2024-07-09 08:13:11.729214","response":"give ","done":false}

...

{"model":"Meta-Llama-3-8B-Instruct","created_at":"2024-07-09 08:13:33.955475","response":"for","done":false}
{"model":"Meta-Llama-3-8B-Instruct","created_at":"2024-07-09 08:13:33.956795","response":"","done":true}
```



