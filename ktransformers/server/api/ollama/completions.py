from datetime import datetime
from http.client import NOT_IMPLEMENTED
import json
from time import time
from uuid import uuid4
from typing import List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ktransformers.server.config.config import Config
from ktransformers.server.utils.create_interface import get_interface
from ktransformers.server.schemas.assistants.streaming import check_link_response
from ktransformers.server.backend.base import BackendInterfaceBase

from ktransformers.server.schemas.endpoints.chat import RawUsage

router = APIRouter(prefix='/api')

# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
class OllamaGenerateCompletionRequest(BaseModel):
    model: str = Field(..., description="The model name, which is required.")
    prompt: Optional[str] = Field(
        None, description="The prompt to generate a response for.")
    images: Optional[List[str]] = Field(
        None, description="A list of base64-encoded images for multimodal models such as llava.")
    # Advanced parameters
    format: Optional[str] = Field(
        None, description="The format to return a response in, accepted value is json.")
    options: Optional[dict] = Field(
        None, description="Additional model parameters as listed in the documentation.")
    system: Optional[str] = Field(
        None, description="System message to override what is defined in the Modelfile.")
    template: Optional[str] = Field(
        None, description="The prompt template to use, overriding what is defined in the Modelfile.")
    context: Optional[str] = Field(
        None, description="The context parameter from a previous request to keep a short conversational memory.")
    stream: Optional[bool] = Field(
        None, description="If false, the response will be returned as a single response object.")
    raw: Optional[bool] = Field(
        None, description="If true, no formatting will be applied to the prompt.")
    keep_alive: Optional[str] = Field(
        "5m", description="Controls how long the model will stay loaded into memory following the request.")

class OllamaGenerationStreamResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool = Field(...)

class OllamaGenerationResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool

@router.post("/generate", tags=['ollama'])
async def generate(request: Request, input: OllamaGenerateCompletionRequest):
    id = str(uuid4())
    interface: BackendInterfaceBase = get_interface()
    print(f'COMPLETION INPUT:----\n{input.prompt}\n----')
    config = Config()

    if input.stream:
        async def inner():
            async for res in interface.inference(input.prompt, id):
                if isinstance(res, RawUsage):
                    raw_usage = res
                else: 
                    token, finish_reason = res
                    d = OllamaGenerationStreamResponse(
                        model=config.model_name,
                        created_at=str(datetime.now()),
                        response=token,
                        done=False
                    )
                    yield d.model_dump_json() + '\n'
            d = OllamaGenerationStreamResponse(
                model=config.model_name,
                created_at=str(datetime.now()),
                response='',
                done=True
            )
            yield d.model_dump_json() + '\n'
        return check_link_response(request, inner())
    else:
        complete_response = ""
        async for res in interface.inference(input.prompt, id):
            if isinstance(res, RawUsage):
                raw_usage = res
            else: 
                token, finish_reason = res
                complete_response += token
        response = OllamaGenerationResponse(
            model=config.model_name,
            created_at=str(datetime.now()),
            response=complete_response,
            done=True
        )
        return response
    
# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
class OllamaChatCompletionMessage(BaseModel):
    role: str
    content: str

class OllamaChatCompletionRequest(BaseModel):
    model: str = Field(..., description="The model name, which is required.")
    messages: List[OllamaChatCompletionMessage] = Field(
        ..., description="A list of messages to generate a response for.")
    stream: bool = Field(True, description="If true, the response will be streamed.")

class OllamaChatCompletionStreamResponse(BaseModel):
    model: str
    created_at: str
    message: dict
    done: bool = Field(...)
    done_reason: Optional[str] = Field("", description="done_reason")
    total_duration: Optional[int] = Field(None, description="Total time spent in nanoseconds")
    load_duration: Optional[int] = Field(None, description="Time spent loading model in nanoseconds")
    prompt_eval_count: Optional[int] = Field(None, description="Number of tokens in prompt")
    prompt_eval_duration: Optional[int] = Field(None, description="Time spent evaluating prompt in nanoseconds")
    eval_count: Optional[int] = Field(None, description="Number of tokens generated")
    eval_duration: Optional[int] = Field(None, description="Time spent generating response in nanoseconds")

class OllamaChatCompletionResponse(BaseModel):
    model: str
    created_at: str
    message: dict
    done: bool
    done_reason: Optional[str] = Field("", description="done_reason")
    total_duration: Optional[int] = Field(None, description="Total time spent in nanoseconds")
    load_duration: Optional[int] = Field(None, description="Time spent loading model in nanoseconds")
    prompt_eval_count: Optional[int] = Field(None, description="Number of tokens in prompt")
    prompt_eval_duration: Optional[int] = Field(None, description="Time spent evaluating prompt in nanoseconds")
    eval_count: Optional[int] = Field(None, description="Number of tokens generated")
    eval_duration: Optional[int] = Field(None, description="Time spent generating response in nanoseconds")

@router.post("/chat", tags=['ollama'])
async def chat(request: Request, input: OllamaChatCompletionRequest):
    id = str(uuid4())
    interface: BackendInterfaceBase = get_interface()
    config = Config()

    input_message = [json.loads(m.model_dump_json()) for m in input.messages]

    if input.stream:
        async def inner():
            start_time = time()  # 记录开始时间（秒）
            tokens = []

            async for res in interface.inference(input_message, id):
                if isinstance(res, RawUsage):
                    raw_usage = res
                else: 
                    token, finish_reason = res
                    d = OllamaChatCompletionStreamResponse(
                        model=config.model_name,
                        created_at=str(datetime.now()),
                        message={"role": "assistant", "content": token}, 
                        done=False
                    )
                    yield d.model_dump_json() + '\n'
            # 计算性能数据
            end_time = time()
            total_duration = int((end_time - start_time) * 1_000_000_000) # unit: ns
            prompt_eval_count = raw_usage.prefill_count
            eval_count = raw_usage.decode_count
            eval_duration = int(raw_usage.decode_time * 1_000_000_000)
            prompt_eval_duration = int(raw_usage.prefill_time * 1_000_000_000)
            load_duration = int(raw_usage.tokenize_time * 1_000_000_000)
            done_reason = finish_reason

            d = OllamaChatCompletionStreamResponse(
                model=config.model_name,
                created_at=str(datetime.now()),
                message={},
                done=True,
                total_duration=total_duration,
                load_duration=load_duration,
                prompt_eval_count=prompt_eval_count,
                prompt_eval_duration=prompt_eval_duration,
                eval_count=eval_count,
                eval_duration=eval_duration,
                done_reason=done_reason
            )
            yield d.model_dump_json() + '\n'
        return check_link_response(request, inner())
    else:
        start_time = time()
        complete_response = ""
        eval_count = 0 

        async for res in interface.inference(input_message, id):
            if isinstance(res, RawUsage):
                raw_usage = res
            else: 
                token, finish_reason = res
                complete_response += token

        end_time = time()
        total_duration = int((end_time - start_time) * 1_000_000_000) # unit: ns
        prompt_eval_count = raw_usage.prefill_count
        eval_count = raw_usage.decode_count
        eval_duration = int(raw_usage.decode_time * 1_000_000_000)
        prompt_eval_duration = int(raw_usage.prefill_time * 1_000_000_000)
        load_duration = int(raw_usage.tokenize_time * 1_000_000_000)
        done_reason = finish_reason


        response = OllamaChatCompletionResponse(
            model=config.model_name,
            created_at=str(datetime.now()),
            message={"role": "assistant", "content": complete_response},
            done=True,
            total_duration=total_duration,
            load_duration=load_duration,
            prompt_eval_count=prompt_eval_count,
            prompt_eval_duration=prompt_eval_duration,
            eval_count=eval_count,
            eval_duration=eval_duration,
            done_reason=done_reason
        )
        return response
    
# https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
class OllamaModel(BaseModel):
    name: str
    modified_at: str
    size: int
    # TODO: fill the rest correctly

# mock ollama
@router.get("/tags", tags=['ollama'])
async def tags():
    config = Config()
    # TODO: fill this correctly, although it does not effect Tabby
    return {"models": [OllamaModel(name=config.model_name, modified_at="123", size=123)]}

class OllamaModelInfo(BaseModel):
    # TODO: fill this correctly
    pass

class OllamaShowRequest(BaseModel):
    name: str = Field(..., description="Name of the model to show")
    verbose: Optional[bool] = Field(
        None, description="If set to true, returns full data for verbose response fields")

class OllamaShowDetial(BaseModel):
    parent_model: str
    format: str
    family: str
    families: List[str]
    parameter_size: str
    quantization_level: str

class OllamaShowResponse(BaseModel):
    modelfile: str
    parameters: str
    template: str
    details: OllamaShowDetial
    model_info: OllamaModelInfo

    class Config:
        protected_namespaces = ()

@router.post("/show", tags=['ollama'])
async def show(request: Request, input: OllamaShowRequest):
    config = Config()
    # TODO: Add more info in config to return, although it does not effect Tabby
    return OllamaShowResponse(
        modelfile="# Modelfile generated by ...",
        parameters=" ",
        template=" ",
        details=OllamaShowDetial(
            parent_model=" ",
            format="gguf",
            family=" ",
            families=[" "],
            parameter_size=" ",
            quantization_level=" "
        ),
        model_info=OllamaModelInfo()
    )
