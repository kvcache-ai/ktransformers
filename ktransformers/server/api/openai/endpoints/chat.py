import json
from time import time
from uuid import uuid4
from fastapi import APIRouter
from fastapi.requests import Request
from ktransformers.server.utils.create_interface import get_interface
from ktransformers.server.schemas.assistants.streaming import chat_stream_response
from ktransformers.server.schemas.endpoints.chat import ChatCompletionCreate,ChatCompletionChunk,ChatCompletionObject
from ktransformers.server.backend.base import BackendInterfaceBase

router = APIRouter()


@router.post('/chat/completions',tags=['openai'])
async def chat_completion(request:Request,create:ChatCompletionCreate):
    id = str(uuid4())

    interface: BackendInterfaceBase = get_interface()
    # input_ids = interface.format_and_tokenize_input_ids(id,messages=create.get_tokenizer_messages())

    input_message = [json.loads(m.model_dump_json()) for m in create.messages]

    if create.stream:
        async def inner():
            chunk = ChatCompletionChunk(id=id,object='chat.completion.chunk',created=int(time()))
            async for token in interface.inference(input_message,id):     
                chunk.set_token(token)
                yield chunk
        return chat_stream_response(request,inner())
    else:
        comp = ChatCompletionObject(id=id,object='chat.completion.chunk',created=int(time()))
        async for token in interface.inference(input_message,id):     
            comp.append_token(token)
        return comp
