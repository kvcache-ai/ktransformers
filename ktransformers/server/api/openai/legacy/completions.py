import json
from time import time
from uuid import uuid4
from fastapi import APIRouter
from fastapi.requests import Request
from ktransformers.server.utils.create_interface import get_interface
from ktransformers.server.schemas.assistants.streaming import stream_response
from ktransformers.server.schemas.legacy.completions import CompletionCreate,CompletionObject
from ktransformers.server.schemas.endpoints.chat import RawUsage

router = APIRouter()

@router.post("/completions",tags=['openai'])
async def create_completion(request:Request,create:CompletionCreate):
    id = str(uuid4())

    interface = get_interface()
    print(f'COMPLETION INPUT:----\n{create.prompt}\n----')

   
    if create.stream:
        async def inner():
            async for res in interface.inference(create.prompt,id,create.temperature,create.top_p):     
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
        async for res in interface.inference(create.prompt,id,create.temperature,create.top_p):     
            if isinstance(res, RawUsage):
                raw_usage = res
            else: 
                token, finish_reason = res
                comp.append_token(token) 
        return comp
