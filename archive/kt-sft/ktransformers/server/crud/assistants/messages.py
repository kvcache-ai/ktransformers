from time import time
from typing import Optional
from uuid import uuid4

from ktransformers.server.models.assistants.messages import Message
from ktransformers.server.schemas.assistants.messages import MessageCore, MessageCreate,  MessageObject
from ktransformers.server.schemas.base import Order,ObjectID
from ktransformers.server.utils.sql_utils import SQLUtil

class MessageDatabaseManager:
    def __init__(self) -> None:
        self.sql_util = SQLUtil()

    @staticmethod
    def create_db_message_by_core(message: MessageCore):
        message_dict = message.model_dump(mode="json")
        return Message(**message_dict, id=str(uuid4()), created_at=int(time()))

    def create_db_message(self, message: MessageCreate):
        return MessageDatabaseManager.create_db_message_by_core(message.to_core())

    def db_add_message(self, message: Message):
        with self.sql_util.get_db() as db:
            db.add(message)
            self.sql_util.db_add_commit_refresh(db, message)

    def db_create_message(self, thread_id: str, message: MessageCreate, status: MessageObject.Status):
        db_message = self.create_db_message(message)
        db_message.status = status.value
        db_message.thread_id = thread_id
        self.db_add_message(db_message)
        return MessageObject.model_validate(db_message.__dict__)

    @staticmethod
    def create_message_object(thread_id: ObjectID, run_id: ObjectID, message: MessageCreate):
        core = message.to_core()
        return MessageObject(
            **core.model_dump(mode='json'),
            id=str(uuid4()),
            object='thread.message',
            created_at=int(time()),
            thread_id=thread_id,
            run_id=run_id,
            status=MessageObject.Status.in_progress,
        )

    def db_sync_message(self, message: MessageObject):
        db_message = Message(
            **message.model_dump(mode="json"),
        )
        with self.sql_util.get_db() as db:
            self.sql_util.db_merge_commit(db, db_message)

    def db_list_messages_of_thread(
            self, thread_id: str, limit: Optional[int] = None, order: Order = Order.DESC):

        # logger.debug(
        #     f"list messages of: {thread_id}, limit {limit}, order {order}")
        with self.sql_util.get_db() as db:
            query = (
                db.query(Message)
                .filter(Message.thread_id == thread_id)
                .order_by(order.to_sqlalchemy_order()(Message.created_at))
            )
            if limit is not None:
                messages = query.limit(limit)
            else:
                messages = query.all()
            message_list = [MessageObject.model_validate(m.__dict__) for m in messages]
        return message_list

    def db_get_message_by_id(self, thread_id: ObjectID, message_id: ObjectID) -> MessageObject:
        with self.sql_util.get_db() as db:
            message = db.query(Message).filter(
                Message.id == message_id).first()
        assert message.thread_id == thread_id
        message_info = MessageObject.model_validate(message.__dict__)
        return message_info

    def db_delete_message_by_id(self, thread_id: ObjectID, message_id: ObjectID):
        with self.sql_util.get_db() as db:
            message = db.query(Message).filter(
                Message.id == message_id).first()
            assert message.thread_id == thread_id
            db.delete(message)
            db.commit()
