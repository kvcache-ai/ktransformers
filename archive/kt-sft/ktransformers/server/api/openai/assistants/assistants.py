from typing import Optional

from fastapi import APIRouter
from fastapi.testclient import TestClient

from ktransformers.server.crud.assistants.assistants import AssistantDatabaseManager
from ktransformers.server.crud.assistants.runs import RunsDatabaseManager
from ktransformers.server.schemas.assistants.assistants import AssistantCreate, AssistantModify, ObjectID, AssistantBuildStatus, AssistantObject
from ktransformers.server.schemas.base import DeleteResponse, Order
from ktransformers.server.config.log import logger


router = APIRouter(prefix="/assistants")
assistant_manager = AssistantDatabaseManager()
runs_manager = RunsDatabaseManager()


@router.post("/", tags=['openai'])
async def create_assistant(
    assistant: AssistantCreate,
):
    return assistant_manager.db_create_assistant(assistant).as_api_response()


@router.get("/", tags=['openai'])
async def list_assistants(
    limit: Optional[int] = 20,
    order: Order = Order.DESC,
    after: Optional[str] = None,
    before: Optional[str] = None,
):
    return [assistant.as_api_response() for assistant in assistant_manager.db_list_assistants(limit, order)]

# list assistant with status


@router.get("/status", tags=['openai-ext'])
async def list_assistants_with_status(
    limit: Optional[int] = 20,
    order: Order = Order.DESC,
    after: Optional[str] = None,
    before: Optional[str] = None,
):
    return assistant_manager.db_list_assistants(limit, order)


@router.get("/{assistant_id}", tags=['openai'])
async def retrieve_assistant(
    assistant_id: str,
):
    return assistant_manager.db_get_assistant_by_id(assistant_id).as_api_response()


@router.post("/{assistant_id}", tags=['openai'])
async def modify_assistant(
    assistant_id: str,
    assistant: AssistantModify,
):
    return assistant_manager.db_update_assistant_by_id(assistant_id, assistant).as_api_response()


@router.delete("/{assistant_id}", tags=['openai'], response_model=DeleteResponse)
async def delete_assistant(assistant_id: str):
    assistant_manager.db_delete_assistant_by_id(assistant_id)
    return DeleteResponse(id=assistant_id, object="assistant.deleted")


@router.get("/{assistant_id}/related_thread", tags=['openai'])
async def get_related_thread(assistant_id: ObjectID):
    assistant = assistant_manager.db_get_assistant_by_id(assistant_id)
    return assistant.get_related_threads_ids()


def create_default_assistant():
    logger.info('Creating default assistant')
    if assistant_manager.db_count_assistants() == 0:
        default_assistant = assistant_manager.db_create_assistant(AssistantCreate(name="KT Assistant",
                                                                                  model="default model",
                                                                                  instructions="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  """ +
                                                                                  """Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. """ +
                                                                                  """Please ensure that your responses are socially unbiased and positive in nature."""))
        default_assistant.build_status.status = AssistantBuildStatus.Status.completed
        default_assistant.sync_db()


# unit test
client = TestClient(router)


def test_create_assistant():
    ass_create = AssistantCreate(model="awesome model", instructions="hello")

    res = client.post("/", json=ass_create.model_dump(mode="json"))

    assert res.status_code == 200
    assistant = AssistantObject.model_validate(res.json())

    assert assistant.model == ass_create.model
    assert assistant.instructions == ass_create.instructions

    res = client.get(f"/{assistant.id}")
    ass1 = AssistantObject.model_validate(res.json())
    assert assistant == ass1
