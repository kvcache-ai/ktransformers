from enum import Enum
from typing import List
from typing_extensions import Self 

from pydantic import BaseModel, Field, model_validator

from ktransformers.server.schemas.base import Metadata, MetadataField, ObjectWithCreatedTime
from ktransformers.server.schemas.assistants.tool import ToolResource
from ktransformers.server.schemas.assistants.messages import MessageCore


class ThreadBase(BaseModel):
    meta_data: Metadata = MetadataField
    @model_validator(mode='before')
    @classmethod
    def convert_meta_data(cls,values):
        if 'meta_data' in values:
            values['metadata'] = values['meta_data']
        return values

    tool_resources: List[ToolResource] = Field([], max_length=128)


class ThreadObject(ThreadBase, ObjectWithCreatedTime):
    is_related_threads:bool = Field(False,exclude=True)

    @model_validator(mode='after')
    def check_is_related_threads(self)->Self:
        # logger.debug(f'check thread {self.id} is related thread? by {self}')
        if 'assistant_id' in self.meta_data:
            self.is_related_threads = True
        return self

    class StreamEvent(Enum):
        created = 'created'

    def to_stream_reply(self,event:StreamEvent):
        return f"event: thread.{event.value}\ndata: {self.model_dump_json()}\n\n"
    

class ThreadCreate(ThreadBase):
    messages: List[MessageCore] = Field(default=[])


class ThreadModify(ThreadBase):
    pass


# other than OpenAI API
