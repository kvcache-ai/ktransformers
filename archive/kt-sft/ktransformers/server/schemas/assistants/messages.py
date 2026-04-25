from enum import Enum
from typing import ForwardRef, List, Optional, Union,Callable

import torch
from pydantic import BaseModel, PrivateAttr, model_validator

from ktransformers.server.exceptions import not_implemented
from ktransformers.server.config.log import logger
from ktransformers.server.models.assistants.messages import Message
from ktransformers.server.schemas.base import Metadata, MetadataField, ObjectWithCreatedTime
from ktransformers.server.schemas.assistants.tool import Field,CodeInterpreter,FileSearch
from ktransformers.server.utils.sql_utils import SQLUtil


class IncompleteDetails(BaseModel):
    reason: str


class ContentType(Enum):
    image_file = "image_file"
    image_url = "image_url"
    text = "text"


class ContentObject(BaseModel):
    type: ContentType


class ImageFile(BaseModel):
    file_id: str
    detail: str


class ImageFileObject(ContentObject):
    image_file: ImageFile


class ImageUrl(BaseModel):
    url: str
    detail: str


class ImageUrlObject(ContentObject):
    image_url: ImageUrl


class Annotation(BaseModel):
    todo: str


class Text(BaseModel):
    value: str
    annotations: List[Annotation] = Field(default=[])


class TextObject(ContentObject):
    text: Text
    delta_index: int = Field(default=0,exclude=True)
    special_tokens_on: bool = Field(default=False,exclude=True) 
    last_two: str= Field(default='',exclude=True)  

    def filter_append(self,text:str):     
        self.text.value+=text
        self.delta_index+=1
        return True  



Content = Union[ImageFileObject, ImageUrlObject, TextObject]


class Attachment(BaseModel):
    file_id: Optional[str] = Field(default=None)
    tools: Optional[List[Union[CodeInterpreter, FileSearch]]] = Field(default=None)


class Role(Enum):
    user = "user"
    assistant = "assistant"

    def is_user(self)->bool:
        return self == Role.user


class MessageCore(BaseModel):
    role: Role
    content: List[Content]
    attachments: Optional[List[Attachment]]
    meta_data: Metadata = MetadataField
    @model_validator(mode='before')
    @classmethod
    def convert_meta_data(cls,values):
        if 'meta_data' in values:
            values['metadata'] = values['meta_data']
        return values


class MessageBase(MessageCore):
    class Status(Enum):
        created = "created" # only used for stream
        in_progress = "in_progress"
        incomplete = "incomplete"
        completed = "completed"
    thread_id: str
    status: Status
    incomplete_details: Optional[IncompleteDetails] = None
    completed_at: Optional[int] = None
    incomplete_at: Optional[int] = None

    assistant_id: Optional[str] = None
    run_id: Optional[str]


MessageStreamResponse = ForwardRef('MessageStreamResponse')

class MessageObject(MessageBase, ObjectWithCreatedTime):
    _encoded_content: Optional[torch.Tensor] = PrivateAttr(default=None)
    

    def get_text_content(self) -> str:
        text_content = ""
        for content in self.content:
            if content.type == ContentType.text:
                text_content += content.text.value
            else:
                raise not_implemented("Content other than text")
        return text_content

    async def get_encoded_content(self,encode_fn:Callable):
        if self._encoded_content is None:
            logger.info(f'encoding {self.role.value} message({self.status.value}): {self.get_text_content()}')
            self._encoded_content = encode_fn(self.get_text_content(),self.role)

            for f in self.get_attached_files():
                logger.info(f'encoding file: {f.filename}')
                self._encoded_content = torch.cat([self._encoded_content, encode_fn(await f.get_str(),self.role)],dim=-1)
                yield None 

        yield self._encoded_content


    def get_attached_files(self):
        raise NotImplementedError # should be replaced 



    def append_message_delta(self,text:str):
        raise NotImplementedError # should be replaced 
    
    def sync_db(self):
        # raise NotImplementedError # should be replaced
        sql_utils = SQLUtil()
        db_message = Message(
            **self.model_dump(mode="json"),
        )
        with sql_utils.get_db() as db:
            sql_utils.db_merge_commit(db, db_message)
    

    def stream_response_with_event(self, event: MessageBase.Status) -> MessageStreamResponse:
        match event:
            case MessageObject.Status.created:
                self.status = MessageObject.Status.in_progress
            case _:
                self.status = event
        return MessageStreamResponse(message=self, event=event)
   

class MessageStreamResponse(BaseModel):
    message: MessageObject
    event: MessageObject.Status

    def to_stream_reply(self):
        return f"event: thread.message.{self.event.value}\ndata: {self.message.model_dump_json()}\n\n"


class MessageCreate(BaseModel):
    role: Role = Field(default=Role.user)
    content: Union[str | List[Content]]
    attachments: Optional[List[Attachment]] = None
    meta_data: Metadata = MetadataField
    @model_validator(mode='before')
    @classmethod
    def convert_meta_data(cls,values):
        if 'meta_data' in values:
            values['metadata'] = values['meta_data']
        return values

    def to_core(self) -> MessageCore:
        # logger.debug(f"Converting message create to core {self.model_dump()}")
        core = MessageCore(
            role=self.role,
            content=[],
            attachments=self.attachments,
            meta_data=self.meta_data,
        )
        if isinstance(self.content, str):
            core.content = [TextObject(type="text", text=Text(value=self.content, annotations=[]))]
        elif isinstance(self.content, list):
            core.content = self.content
        else:
            raise ValueError("Invalid content type")
        return core


class MessageModify(BaseModel):
    meta_data: Metadata = MetadataField
    @model_validator(mode='before')
    @classmethod
    def convert_meta_data(cls,values):
        if 'meta_data' in values:
            values['metadata'] = values['meta_data']
        return values
