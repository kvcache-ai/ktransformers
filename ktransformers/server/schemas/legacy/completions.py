from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field

from ..base import Object

class CompletionCreate(BaseModel):
    model: str
    prompt: str | List[str]
    stream: bool = False
    temperature: Optional[float] = Field(default=0.6)
    top_p: Optional[float] = Field(default=1)
    max_tokens: Optional[int] = Field(default=50)
    max_completion_tokens: Optional[int] = Field(default=50)
    
    def get_tokenizer_messages(self):
        if isinstance(self.prompt,List):
            self.get_tokenizer_messages('\n'.join(self.prompt))
        return [{'content':self.prompt,'role':'user'}]


class FinishReason(Enum):
    stop = 'stop'
    length = 'length'

class Choice(BaseModel):
    index: int
    text: str
    logprobs: Optional[str] = None
    finish_reason: FinishReason = None


class CompletionObject(Object):
    created:int
    choices: List[Choice] = []
    model:str = 'not implmented'
    system_fingerprint:str = 'not implmented'
    usage: Optional[str] = None

    def set_token(self,token:str):
        if len(self.choices)==0:
            self.choices.append(Choice(index=0,text=''))
        self.choices[0].text = token    

    def append_token(self,token:str):
        if len(self.choices)==0:
            self.choices.append(Choice(index=0,text=''))
        self.choices[0].text += token

    def to_stream_reply(self):
        return f"data:{self.model_dump_json()}\n\n"
