import json
from time import time
from uuid import uuid4

from fastapi import APIRouter
from fastapi.requests import Request

from ktransformers.server.utils.create_interface import get_interface
from ktransformers.server.schemas.assistants.streaming import chat_stream_response
from ktransformers.server.schemas.endpoints.chat import ChatCompletionCreate
from ktransformers.server.schemas.endpoints.chat import RawUsage
from ktransformers.server.backend.base import BackendInterfaceBase
from ktransformers.server.config.config import Config

from ktransformers.server.schemas.endpoints.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletion
from openai.types.completion_usage import  CompletionUsage


router = APIRouter()


@router.get('/models', tags=['openai'])
async def list_models():
    return {"data": [{"id": Config().model_name, "name": Config().model_name}], "object": "list"}


@router.post('/chat/completions', tags=['openai'])
async def chat_completion(request:Request,create:ChatCompletionCreate):
    id = str(uuid4())

    interface: BackendInterfaceBase = get_interface()
    # input_ids = interface.format_and_tokenize_input_ids(id,messages=create.get_tokenizer_messages())

    input_message = [json.loads(m.model_dump_json()) for m in create.messages]

    if Config().api_key != '':
        assert request.headers.get('Authorization', '').split()[-1] == Config().api_key

    if create.stream:

        from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

        async def inner():
            chunk = ChatCompletionChunk(
                id = id,
                choices = [],
                object = 'chat.completion.chunk',
                created = int(time()),
                model = Config().model_name,
            )

            async for res in interface.inference(input_message, id, create.temperature, create.top_p, create.max_tokens, create.max_completion_tokens):
                if isinstance(res, RawUsage):
                    # at the end of inference, inference.inference() will return the usage of inference
                    raw_usage = res
                    chunk.choices = []
                    chunk.usage = CompletionUsage(
                        prompt_tokens = raw_usage.prefill_count,
                        completion_tokens = raw_usage.decode_count,
                        total_tokens = raw_usage.prefill_count + raw_usage.decode_count,
                    )

                    yield chunk

                else:
                    token, finish_reason = res
                    choice = Choice(
                        index = 0,
                        delta = ChoiceDelta(content=token, role=None, tool_calls=None),
                        finish_reason = finish_reason,
                        logprobs = None,
                    )
                    chunk.choices = [choice]
                    yield chunk

        return chat_stream_response(request, inner())
    else:
        from openai.types.chat.chat_completion import Choice
        from openai.types.chat.chat_completion_message import ChatCompletionMessage

        content = ""
        finish_reason = None
        async for res in interface.inference(input_message, id, create.temperature, create.top_p):
            if isinstance(res, RawUsage):
                raw_usage = res
                usage = CompletionUsage(
                    prompt_tokens = raw_usage.prefill_count,
                    completion_tokens = raw_usage.decode_count,
                    total_tokens = raw_usage.prefill_count + raw_usage.decode_count
                )
            else:
                token, finish_reason = res
                content = content + token
                finish_reason = finish_reason

            choice = Choice(
                index = 0,
                finish_reason = finish_reason,
                message = ChatCompletionMessage(
                    content=content, 
                    role="assistant"
                ))

            chat_completion = ChatCompletion(
                id = id,
                choices = [choice],
                created = int(time()),
                model = Config().model_name,
                object = 'chat.completion',
                usage = usage,
            )

            return chat_completion
