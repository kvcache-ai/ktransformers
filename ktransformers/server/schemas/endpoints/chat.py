from typing import List, Optional
from enum import Enum

from pydantic import BaseModel

from ktransformers.server.schemas.base import Object

class Role(Enum):
    system = 'system'
    user = 'user'
    assistant = 'assistant'
    tool = 'tool'
    function = 'function'


class Message(BaseModel):
    content: str
    role:Role
    name: Optional[str] = None
    def to_tokenizer_message(self):
        return {'content':self.content,'role':self.role.value}


class ChatCompletionCreate(BaseModel):
    messages: List[Message]
    model : str
    stream : bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    
    def get_tokenizer_messages(self):
        return [m.to_tokenizer_message() for m in self.messages]

class FinishReason(Enum):
    stop = 'stop'
    length = 'length'

class Choice(BaseModel):
    index: int
    message: Message
    logprobs: Optional[str] = None
    finish_reason: FinishReason = None

class DeltaChoice(BaseModel):
    index: int
    delta: Message
    logprobs: Optional[str] = None
    finish_reason: FinishReason = None


class Usage(BaseModel):
    completion_tokens:int
    prompt_tokens:int
    total_tokens:int


class ChatCompletionBase(Object):
    created:int
    model:str = 'not implmented'
    system_fingerprint:str = 'not implmented'
    usage: Optional[Usage] = None

class ChatCompletionObject(ChatCompletionBase):
    choices:List[Choice] = []

    def append_token(self,token:str):
        if len(self.choices) == 0:
            self.choices.append(Choice(index=0,message=Message(content='',role=Role.assistant)))
        self.choices[0].message.content += token

class ChatCompletionChunk(ChatCompletionBase):
    choices:List[DeltaChoice] = []

    def set_token(self,token:str):
        self.choices = [
            DeltaChoice(index=0,delta=Message(content=token,role=Role.assistant))
        ]

    def to_stream_reply(self):
        return f"data: {self.model_dump_json()}\n\n"
