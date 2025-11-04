from typing import List,Optional
from fastapi import APIRouter

from ktransformers.server.crud.assistants.threads import ThreadsDatabaseManager,Order,ObjectID
from ktransformers.server.schemas.assistants.threads import ThreadObject,ThreadCreate,ThreadModify
from ktransformers.server.schemas.base import DeleteResponse
from ktransformers.server.schemas.conversation import ThreadPreview

router = APIRouter(prefix='/threads')
threads_manager = ThreadsDatabaseManager()


@router.post("/",tags=['openai'], response_model=ThreadObject)
async def create_thread(thread: ThreadCreate):
    return threads_manager.db_create_thread(thread)


@router.get("/", tags=['openai-ext'],response_model=List[ThreadPreview])
async def list_threads(limit: Optional[int] = 20, order: Order = Order.DESC):
    return threads_manager.db_list_threads_preview(limit, order)


@router.get("/{thread_id}",tags=['openai'], response_model=ThreadObject)
async def retrieve_thread(thread_id: ObjectID):
    return threads_manager.db_get_thread_by_id(thread_id)


@router.post("/{thread_id}",tags=['openai'], response_model=ThreadObject)
async def modify_thread(thread_id: ObjectID, thread: ThreadModify):
    raise NotImplementedError


@router.delete("/{thread_id}",tags=['openai'], response_model=DeleteResponse)
async def delete_thread(thread_id: ObjectID):
    threads_manager.db_delete_thread_by_id(thread_id=thread_id)
    return DeleteResponse(id=thread_id, object='thread.deleted')
