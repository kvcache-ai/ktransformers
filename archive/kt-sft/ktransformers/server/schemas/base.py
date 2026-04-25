from enum import Enum
from typing import Dict

import sqlalchemy
from pydantic import BaseModel, ConfigDict, Field

TODO = BaseModel

ObjectID = str


class Object(BaseModel):
    id: ObjectID
    object: str

    model_config = ConfigDict(from_attributes=True)


# Pydantic Base Models
class ObjectWithCreatedTime(Object):
    created_at: int



class Order(str, Enum):
    ASC = "asc"
    DESC = "desc"

    def to_sqlalchemy_order(self):
        match self:
            case Order.ASC:
                return sqlalchemy.asc
            case Order.DESC:
                return sqlalchemy.desc


Metadata = Dict[str, str]
MetadataField: Metadata = Field({},max_length=16, alias="metadata")


class DeleteResponse(Object):
    deleted: bool = True

class OperationResponse(BaseModel):
    operation: str
    status: str
