from enum import Enum
from typing import Dict, List, Optional, Union, ForwardRef

from pydantic import BaseModel, Field, model_validator

from ktransformers.server.models.assistants.runs import Run
from ktransformers.server.schemas.base import TODO, Metadata, MetadataField, ObjectWithCreatedTime
from ktransformers.server.schemas.assistants.threads import ThreadCreate
from ktransformers.server.schemas.assistants.tool import Tool, ToolResource
from ktransformers.server.utils.sql_utils import SQLUtil


class ToolCall(BaseModel):
    id: str
    type: str
    function: TODO


class SubmitToolOutputs(BaseModel):
    tool_calls: List[ToolCall]


class RequiredAction(BaseModel):
    type: str
    submit_tool_outputs: TODO


class LastError(BaseModel):
    code: str
    message: str


class IncompleteDetails(BaseModel):
    reason: str


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class TruncationStrategy(BaseModel):
    type: str = "auto"
    last_message: Optional[int]


class ToolChoiceType(Enum):
    none = "none"
    auto = "auto"
    required = "required"


class RunBase(BaseModel):
    class Status(Enum):
        created = "created" # only stream event will have this created status
        queued = "queued"
        in_progress = "in_progress"
        requires_action = "requires_action"
        cancelling = "cancelling"
        cancelled = "cancelled"
        failed = "failed"
        completed = "completed"
        expired = "expired"


    thread_id: str
    assistant_id: str
    status: Status = Status.queued
    required_action: Optional[RequiredAction] = Field(None)
    last_error: Optional[LastError] = Field(None)
    expires_at: Optional[int]= Field(None)
    started_at: Optional[int] = Field(None)
    cancelled_at: Optional[int] = Field(None)
    failed_at: Optional[int] = Field(None)
    completed_at: Optional[int] = Field(None)
    incomplete_details: Optional[IncompleteDetails] = Field(None)
    model: Optional[str] = Field(None)
    instructions: Optional[str] = Field(None)
    tools: Optional[List[Tool]] = Field([])
    meta_data: Metadata = MetadataField
    @model_validator(mode='before')
    @classmethod
    def convert_meta_data(cls,values):
        if 'meta_data' in values:
            values['metadata'] = values['meta_data']
        return values
    
    def set_compute_save(self,save:int):
        self.meta_data['compute_save'] = str(save)


    usage: Optional[Usage] = Field(None)
    temperature: Optional[float] = Field(None)
    top_p: Optional[float]= Field(None)
    max_propmp_tokens: Optional[int]= Field(None)
    truncation_strategy: Optional[TruncationStrategy]= Field(None)
    tool_choice: Optional[Union[ToolChoiceType, dict]]= Field(None)
    response_format: Union[str, Dict[str, str]] = "auto"


RunStreamResponse = ForwardRef('RunStreamResponse')

class RunObject(RunBase, ObjectWithCreatedTime):
    def stream_response_with_event(self,event:RunBase.Status)->RunStreamResponse:
        match event:
            case RunBase.Status.created:
                self.status = RunBase.Status.queued
            case _:
                self.status = event
        return RunStreamResponse(run=self, event=event)
 
    
    def sync_db(self):
        # raise NotImplementedError # should be replaced in crud
        sql_utils = SQLUtil()
        db_run = Run(
            **self.model_dump(mode='json'),
        )
        with sql_utils.get_db() as db:
            sql_utils.db_merge_commit(db, db_run)
    
    def create_message_creation_step(self):
        raise NotImplementedError # should be replaced 
        

class RunStreamResponse(BaseModel):
    run: RunObject
    event: RunObject.Status
    def to_stream_reply(self):
        return f"event: thread.run.{self.event.value}\ndata: {self.run.model_dump_json()}\n\n"

class RunCreate(BaseModel):
    assistant_id: str
    model: Optional[str] = Field(default=None)
    instructions: Optional[str] = Field(default=None)
    # TODO: Add this
    # additional_instructions: Optional[str]
    # additional_messages: Optional[List[MessageCore]]
    tools: List[Tool] = Field(default=[])
    meta_data: Metadata = MetadataField
    @model_validator(mode='before')
    @classmethod
    def convert_meta_data(cls,values):
        if 'meta_data' in values:
            values['metadata'] = values['meta_data']
        return values
    temperature: Optional[float] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    stream: Optional[bool] = Field(default=None)
    max_propmp_tokens: Optional[int] = Field(default=None)
    # TODO: Add this
    # max_completion_tokens: Optional[int]
    truncation_strategy: Optional[TruncationStrategy] = Field(default=None)
    tool_choice: Optional[Union[ToolChoiceType, dict]] = Field(default=None)
    response_format: Union[str, Dict[str, str]] = Field(default="auto")


class RunThreadCreate(BaseModel):
    assistant_id: str
    thread: Optional[ThreadCreate]
    model: Optional[str]
    instructions: Optional[str]
    tools: List[Tool]
    tool_resources: List[ToolResource]
    meta_data: Metadata = MetadataField
    @model_validator(mode='before')
    @classmethod
    def convert_meta_data(cls,values):
        if 'meta_data' in values:
            values['metadata'] = values['meta_data']
        return values
    temperature: Optional[float]
    top_p: Optional[float]
    stream: Optional[bool]
    max_propmp_tokens: Optional[int]
    # TODO: Add this
    # max_completion_tokens: Optional[int]
    truncation_strategy: TruncationStrategy
    tool_choice: Union[ToolChoiceType, dict]
    response_format: Union[str, Dict[str, str]] = "auto"


class RunModify(BaseModel):
    meta_data: Metadata = MetadataField
    @model_validator(mode='before')
    @classmethod
    def convert_meta_data(cls,values):
        if 'meta_data' in values:
            values['metadata'] = values['meta_data']
        return values


class ToolOutput(BaseModel):
    tool_call_id: Optional[str]
    output: Optional[str]


class RunSubmit(BaseModel):
    tool_outputs: List[ToolOutput]
    stream: Optional[bool]
