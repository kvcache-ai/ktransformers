import json
from time import time
from uuid import uuid4
from fastapi import APIRouter
from fastapi.requests import Request
from ktransformers.server.utils.create_interface import get_interface
from ktransformers.server.schemas.assistants.streaming import chat_stream_response
from ktransformers.server.schemas.endpoints.chat import ChatCompletionCreate,ChatCompletionChunk,ChatCompletionObject, Usage
from ktransformers.server.backend.base import BackendInterfaceBase
from ktransformers.server.config.config import Config

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
        async def inner():
            chunk = ChatCompletionChunk(id=id,object='chat.completion.chunk',created=int(time()))
            async for token in interface.inference(input_message,id,create.temperature,create.top_p):
                chunk.set_token(token)
                yield chunk
        return chat_stream_response(request,inner())
    else:
        comp = ChatCompletionObject(id=id,object='chat.completion',created=int(time()))
        comp.usage = Usage(completion_tokens=1, prompt_tokens=1, total_tokens=2)
        async for token in interface.inference(input_message,id,create.temperature,create.top_p):
            comp.append_token(token)
        return comp
