import json
from time import time
from uuid import uuid4
from fastapi import APIRouter
from fastapi.requests import Request
from ktransformers.server.utils.create_interface import get_interface
from ktransformers.server.schemas.assistants.streaming import stream_response
from ktransformers.server.schemas.legacy.completions import CompletionCreate,CompletionObject
from ktransformers.server.schemas.endpoints.chat import RawUsage
from fastapi.responses import JSONResponse
from ktransformers.server.config.config import Config
router = APIRouter()

@router.post("/completions",tags=['openai'])
async def create_completion(request:Request, create:CompletionCreate):
    id = str(uuid4())
    if create.max_tokens is not None and create.max_tokens<0:
        return JSONResponse(
            status_code=400,
            content={
            "object": "error",
            "message": f"max_tokens must be at least 0, got {create.max_tokens}.",
            "type": "BadRequestError",
            "param": None,
            "code": 400
        })
    if create.max_completion_tokens is not None and create.max_completion_tokens<0:
        return JSONResponse(
            status_code=400,
            content={
            "object": "error",
            "message": f"max_completion_tokens must be at least 0, got {create.max_completion_tokens}.",
            "type": "BadRequestError",
            "param": None,
            "code": 400
        })
    if create.temperature<0 or create.temperature>2:
        return JSONResponse(
            status_code=400,
            content={
            "object": "error",
            "message": f"temperature must be in [0, 2], got {create.temperature}.",
            "type": "BadRequestError",
            "param": None,
            "code": 400
            })
    if create.top_p<=0 or create.top_p>1:
        return JSONResponse(
            status_code=400,
            content={
            "object": "error",
            "message": f"top_p must be in (0, 1], got {create.top_p}.",
            "type": "BadRequestError",
            "param": None,
            "code": 400
        })
    interface = get_interface()
    print(f'COMPLETION INPUT:----\n{create.prompt}\n----')

   
    if create.stream:
        async def inner():
            async for res in interface.inference(create.prompt, id, create.temperature, create.top_p, create.max_tokens, create.max_completion_tokens):     
                if isinstance(res, RawUsage):
                    raw_usage = res
                else: 
                    token, finish_reason = res
                    d = {'choices':[{'delta':{'content':token}}]}
                    yield f"data:{json.dumps(d)}\n\n"
            d = {'choices':[{'delta':{'content':''},'finish_reason':''}]}
            yield f"data:{json.dumps(d)}\n\n"
        return stream_response(request,inner())
    else:
        comp = CompletionObject(id=id,object='text_completion',created=int(time()))
        async for res in interface.inference(create.prompt,id,create.temperature,create.top_p, create.max_tokens, create.max_completion_tokens):     
            if isinstance(res, RawUsage):
                raw_usage = res
            else: 
                token, finish_reason = res
                comp.append_token(token) 
        return comp
