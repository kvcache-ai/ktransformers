from typing import List, Optional

from fastapi import APIRouter

from ktransformers.server.exceptions import not_implemented
from ktransformers.server.schemas.assistants.messages import MessageCreate, MessageObject, MessageModify
from ktransformers.server.crud.assistants.messages import MessageDatabaseManager
from ktransformers.server.schemas.base import DeleteResponse, ObjectID, Order
from ktransformers.server.backend.base import ThreadContext
from ktransformers.server.utils.create_interface import  get_thread_context_manager
router = APIRouter()
message_manager = MessageDatabaseManager()


@router.post("/{thread_id}/messages", tags=['openai'], response_model=MessageObject)
async def create_message(thread_id: str, msg: MessageCreate):
    message = message_manager.db_create_message(
        thread_id, msg, MessageObject.Status.in_progress)
    ctx: Optional[ThreadContext] = await get_thread_context_manager().get_context_by_thread_id(thread_id)
    if ctx is not None:
        ctx.put_user_message(message)
    return message


@router.get("/{thread_id}/messages", tags=['openai'], response_model=List[MessageObject])
async def list_messages(
    thread_id: str,
    limit: Optional[int] = 20,
    order: Order = Order.DESC,
    after: Optional[str] = None,
    before: Optional[str] = None,
    run_id: Optional[str] = None,
):
    return message_manager.db_list_messages_of_thread(thread_id, limit, order)


@router.get("/{thread_id}/messages/{message_id}", tags=['openai'], response_model=MessageObject)
async def retrieve_message(thread_id: ObjectID, message_id: ObjectID):
    return message_manager.db_get_message_by_id(thread_id, message_id)


@router.post("/{thread_id}/messages/{message_id}", tags=['openai'], response_model=MessageObject)
async def modify_message(thread_id: ObjectID, message_id: ObjectID, msg: MessageModify):
    #raise not_implemented('modify message not implemented')
    raise not_implemented('modify message')


@router.delete("/{thread_id}/messages/{message_id}", tags=['openai'], response_model=DeleteResponse)
async def delete_message(thread_id: ObjectID, message_id: ObjectID):
    ctx: Optional[ThreadContext] = await get_thread_context_manager().get_context_by_thread_id(thread_id)
    if ctx is not None:
        ctx.delete_user_message(message_id)
    message_manager.db_delete_message_by_id(thread_id, message_id)
    return DeleteResponse(id=message_id, object='thread.message.deleted')
