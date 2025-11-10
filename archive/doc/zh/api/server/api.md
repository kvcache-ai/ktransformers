# API


- [OpenAI ChatCompletion](#openai-chatcompletion)
- [Ollama ChatCompletion](#ollama-chatcompletion)
- [OpenAI Assistant](#openai-assistant)


## OpenAI ChatCompletion
```bash
POST /v1/chat/completions
```
根据选定的模型生成回复。

### 参数


- `messages`：一个 `message` 的数组所有的历史消息。`message`：表示用户（user）或者模型（assistant）的消息。`message`包含：

  - `role`: 取值`user`或`assistant`，代表这个 message 的创建者。
  - `content`: 用户或者模型的消息。

- `model`：选定的模型名
- `stream`：取值 true 或者 false。表示是否使用流式返回。如果为 true，则以 http 的 event stream 的方式返回模型推理结果。

### 响应

- 流式返回：一个 event stream，每个 event 含有一个`chat.completion.chunk`。`chunk.choices[0].delta.content`是每次模型返回的增量输出。
- 非流式返回：还未支持。

### 例子

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

根据选定的模型生成回复。

### 参数


- `prompt`：一个字符串，代表输入的 prompt。
- `model`：选定的模型名
- `stream`：取值 true 或者 false。表示是否使用流式返回。如果为 true，则以 http 的 event stream 的方式返回模型推理结果。

### 响应

- 流式返回：一个流式的 json 返回，每行是一个 json。
  - `response`：模型补全的增量结果。
  - `done`：是否推理结束。

- 非流式返回：还未支持。

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



