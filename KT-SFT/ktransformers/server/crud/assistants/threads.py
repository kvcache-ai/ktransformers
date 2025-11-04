from time import time
from typing import Optional,List
from uuid import uuid4

from ktransformers.server.models.assistants.messages import Message
from ktransformers.server.models.assistants.threads import Thread
from ktransformers.server.schemas.assistants.threads import ThreadCreate,ThreadObject
from ktransformers.server.schemas.base import ObjectID, Order
from ktransformers.server.schemas.conversation import ThreadPreview
from ktransformers.server.utils.sql_utils import SQLUtil
from ktransformers.server.crud.assistants.messages import MessageDatabaseManager
from ktransformers.server.config.log import logger
from ktransformers.server.crud.assistants.assistants import AssistantDatabaseManager

class ThreadsDatabaseManager:
    def __init__(self) -> None:
        self.sql_util = SQLUtil()
        self.message_manager = MessageDatabaseManager()
        self.assistant_maanager = AssistantDatabaseManager()

    def db_create_thread(self, thread: ThreadCreate):
        thread_id = str(uuid4())
        db_messages = []
        with self.sql_util.get_db() as db:
            if thread.messages is not None:
                logger.debug("Creating messages first for thread")
                for message in thread.messages:
                    db_message: Message = MessageDatabaseManager.create_db_message_by_core(
                        message)
                    db_message.role = "user"
                    db_message.thread_id = thread_id
                    db.add(db_message)
                    db_messages.append(db_message)

            db_thread = Thread(
                **thread.model_dump(exclude="messages"),
                id=str(uuid4()),
                created_at=int(time()),
                messages=db_messages,
            )

            self.sql_util.db_add_commit_refresh(db, db_thread)
            thread_obj = ThreadObject.model_validate(db_thread.__dict__)

            if 'assistant_id' in thread.meta_data:
#                assistant = self.assistant_maanager.db_get_assistant_by_id(thread.meta_data['assistant_id'], db)
                assistant = self.assistant_maanager.db_get_assistant_by_id(thread.meta_data['assistant_id'])
                logger.info(
                    f'Append this related thread to assistant {assistant.id}')
                assistant.append_related_threads([thread_obj.id])
                assistant.sync_db(db)
        return thread_obj

    def db_get_thread_by_id(self, thread_id: ObjectID):
        with self.sql_util.get_db() as db:
            db_thread = db.query(Thread).filter(Thread.id == thread_id).first()
            return ThreadObject.model_validate(db_thread.__dict__)

    def db_list_threads(self, limit: Optional[int], order: Order) -> List[ThreadObject]:
        with self.sql_util.get_db() as db:
            query = db.query(Thread).order_by(order.to_sqlalchemy_order()(
                Thread.created_at)).filter(~Thread.meta_data.contains('assistant_id'))

            if limit is not None:
                db_threads = query.limit(limit)
            else:
                db_threads = query.all()

            return [ThreadObject.model_validate(tool.__dict__) for tool in db_threads]

    def db_list_threads_preview(self, limit: Optional[int], order: Order) -> List[ThreadPreview]:
        threads = self.db_list_threads(limit, order)
        previews = []
        for thread in threads:
            messages = self.message_manager.db_list_messages_of_thread(
                thread.id, limit=2, order=Order.ASC)
            if len(messages) == 2:
                message = messages[0]
                assistant = self.assistant_maanager.db_get_assistant_by_id(
                    messages[1].assistant_id)
            else:
                message = None
                assistant = None
            previews.append(ThreadPreview(
                assistant=assistant, thread=thread, first_message=message))
        return previews

    def db_delete_thread_by_id(self, thread_id: ObjectID):
        with self.sql_util.get_db() as db:
            db_thread = db.query(Thread).filter(Thread.id == thread_id).first()
            db.delete(db_thread)
            # TODO delete related messages and runs and other stuff or just gc
            db.commit()
