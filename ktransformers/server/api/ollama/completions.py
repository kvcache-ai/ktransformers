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
    pass


@router.post("/generate", tags=['ollama'])
async def generate(request: Request, input: OllamaGenerateCompletionRequest):
    id = str(uuid4())

    interface: BackendInterfaceBase = get_interface()
    print(f'COMPLETION INPUT:----\n{input.prompt}\n----')

    config = Config()

    if input.stream:
        async def inner():
            async for token in interface.inference(input.prompt,id): 
                d = OllamaGenerationStreamResponse(model=config.model_name,created_at=str(datetime.now()),response=token,done=False)
                yield d.model_dump_json()+'\n' 
                # d = {'model':config.model_name,'created_at':"", 'response':token,'done':False}
                # yield f"{json.dumps(d)}\n"
            # d = {'model':config.model_name,'created_at':"", 'response':'','done':True}
            # yield f"{json.dumps(d)}\n"
            d = OllamaGenerationStreamResponse(model=config.model_name,created_at=str(datetime.now()),response='',done=True)   
            yield d.model_dump_json()+'\n'
        return check_link_response(request,inner())
    else:
        raise NotImplementedError

# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion


class OllamaChatCompletionRequest(BaseModel):
    pass


class OllamaChatCompletionStreamResponse(BaseModel):
    pass


class OllamaChatCompletionResponse(BaseModel):
    pass


@router.post("/chat", tags=['ollama'])
async def chat(request: Request, input: OllamaChatCompletionRequest):
    raise NotImplementedError


# https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
class OllamaModel(BaseModel):
    name: str
    modified_at: str
    size: int
    # TODO: fill the rest correctly


# mock ollama
@router.get("/tags",tags=['ollama'])
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
        modelfile = "# Modelfile generated by ...",
        parameters = " ",
        template = " ",
        details = OllamaShowDetial(
            parent_model = " ",
            format = "gguf",
            family = " ",
            families = [
                " " 
            ],
            parameter_size = " ",
            quantization_level = " "
        ),
        model_info = OllamaModelInfo()
    )