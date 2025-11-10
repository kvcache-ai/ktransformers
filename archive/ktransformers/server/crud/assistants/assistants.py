from time import time
from typing import Optional,List
from uuid import uuid4

from ktransformers.server.models.assistants.assistants import Assistant
from ktransformers.server.schemas.assistants.assistants import AssistantCreate,AssistantObject,AssistantModify
from ktransformers.server.utils.sql_utils import SQLUtil
from ktransformers.server.config.log import logger
from ktransformers.server.schemas.base import Order


class AssistantDatabaseManager:
    def __init__(self) -> None:
        self.sql_util = SQLUtil()

    def create_assistant_object(self, assistant: AssistantCreate) -> AssistantObject:
        assistant = AssistantObject(
            **assistant.model_dump(mode='json'),
            id=str(uuid4()),
            object='assistant',
            created_at=int(time()),
        )
        return assistant

    def db_count_assistants(self) -> int:
        with self.sql_util.get_db() as db:
            return db.query(Assistant).count()

    def db_create_assistant(self, assistant: AssistantCreate):
        ass_obj = self.create_assistant_object(assistant)
        ass_obj.sync_db()
        return ass_obj

    def db_list_assistants(self, limit: Optional[int], order: Order) -> List[AssistantObject]:
        with self.sql_util.get_db() as db:
            query = db.query(Assistant).order_by(
                order.to_sqlalchemy_order()(Assistant.created_at))
            if limit is not None:
                db_assistants = query.limit(limit)
            else:
                db_assistants = query.all()
            return [AssistantObject.model_validate(a.__dict__) for a in db_assistants]

    def db_get_assistant_by_id(self, assistant_id: str) -> Optional[AssistantObject]:
        with self.sql_util.get_db() as db:
            db_assistant = db.query(Assistant).filter(
                Assistant.id == assistant_id).first()
            if db_assistant is None:
                logger.debug(f"no assistant with id {str}")
                return None
            return AssistantObject.model_validate(db_assistant.__dict__)

    def db_update_assistant_by_id(self, assistant_id: str, assistant: AssistantModify):
        with self.sql_util.get_db() as db:
            db_assistant = db.query(Assistant).filter(
                Assistant.id == assistant_id).first()
            self.sql_util.db_update_commit_refresh(db, db_assistant, assistant)
            return AssistantObject.model_validate(db_assistant.__dict__)

    def db_delete_assistant_by_id(self, assistant_id: str):
        with self.sql_util.get_db() as db:
            db_assistant = db.query(Assistant).filter(
                Assistant.id == assistant_id).first()
            db.delete(db_assistant)
            db.commit()

