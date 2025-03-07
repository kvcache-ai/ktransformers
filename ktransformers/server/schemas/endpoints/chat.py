from typing import List, Optional
from typing_extensions import Literal
from enum import Enum

from pydantic import BaseModel

from ktransformers.server.schemas.base import Object

from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_chunk import Choice


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


class ChatCompletionChunk(BaseModel):
    id: str
    choices: List[Choice]
    created: int
    model: str
    object: Literal["chat.completion.chunk"]
    service_tier: Optional[Literal["scale", "default"]] = None
    system_fingerprint: Optional[str] = None
    usage: Optional[CompletionUsage] = None


    def to_stream_reply(self):
        return f"data: {self.model_dump_json()}\n\n"


class RawUsage(BaseModel):
    tokenize_time: float
    prefill_time: float
    decode_time: float
    prefill_count: int
    decode_count: int
