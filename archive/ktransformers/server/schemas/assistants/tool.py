from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from ktransformers.server.schemas.base import ObjectID


class ToolType(str, Enum):
    CODE_INTERPRETER = "code_interpreter"
    FILE_SEARCH = "file_search"
    RELATED_THREADS = "related_threads"
    FUNCTION = "function"


class ToolBase(BaseModel):
    type: ToolType


class CodeInterpreter(ToolBase):
    pass


class FileSearch(ToolBase):
    pass


class RelatedThreads(ToolBase):
    pass


class FuntionTool(ToolBase):
    description: str
    name: str
    parameters: List[str]


Tool = Union[CodeInterpreter, FileSearch, RelatedThreads, FuntionTool]


class CodeInterpreterResource(BaseModel):
    file_ids: Optional[List[str]] = Field(default_factory=list, max_length=20)


class FileSearchResource(BaseModel):
    vector_store_ids: Optional[List[str]] = Field(default_factory=list, max_length=1)
    vector_stores: Optional[List[str]] = Field(default_factory=list, max_length=1)


class RelatedThreadsResource(BaseModel):
    thread_ids: List[ObjectID] = Field(default=[])


ToolResource = Union[CodeInterpreterResource,FileSearchResource,RelatedThreadsResource] 
