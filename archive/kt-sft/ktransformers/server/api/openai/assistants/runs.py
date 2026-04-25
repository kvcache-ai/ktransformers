from typing import List, Optional

from fastapi import APIRouter, Request

from ktransformers.server.crud.assistants.runs import RunsDatabaseManager
from ktransformers.server.backend.base import ThreadContext
from ktransformers.server.schemas.assistants.runs import RunCreate,RunObject,RunThreadCreate,RunModify,RunSubmit
from ktransformers.server.schemas.assistants.streaming import api_stream_response
from ktransformers.server.utils.create_interface import  get_thread_context_manager
from ktransformers.server.schemas.base import Order
from ktransformers.server.config.log import logger
from ktransformers.server.exceptions import internal_server_error


router = APIRouter()
runs_manager = RunsDatabaseManager()


@router.post("/{thread_id}/runs",tags=['openai'])
async def create_run(request: Request, thread_id: str, run_create: RunCreate):
    if run_create.stream:
        async def inner():
            run = runs_manager.db_create_run(thread_id, run_create)
            yield run.stream_response_with_event(event=RunObject.Status.created)

            ctx: ThreadContext = await get_thread_context_manager().get_context_by_run_object(run)
           
            async for event in ctx.work():
                yield event
        return api_stream_response(request, inner())
    else:
        run = runs_manager.db_create_run(thread_id, run_create)
        ctx: ThreadContext = await get_thread_context_manager().get_context_by_run_object(run)
        async for event in ctx.work():
            pass
        return run


@router.post("/runs",tags=['openai'], response_model=RunObject)
async def create_thread_and_run(run_thread: RunThreadCreate):
    raise NotImplementedError


@router.get("/{thread_id}/runs",tags=['openai'], response_model=List[RunObject])
async def list_runs(
    thread_id: str,
    limit: Optional[int] = 20,
    order: Optional[Order] = Order.DESC,
    after: Optional[str] = None,
    before: Optional[str] = None,
):
    raise NotImplementedError


@router.get("/{thread_id}/runs/{run_id}",tags=['openai'], response_model=RunObject)
async def retrieve_run(
    thread_id: str,
    run_id: str,
):
    runobj= runs_manager.db_get_run(run_id)
    assert runobj.thread_id == thread_id
    return runobj



@router.post("/{thread_id}/runs/{run_id}",tags=['openai'], response_model=RunObject)
async def modify_run(
    thread_id: str,
    run_id: str,
    run: RunModify,
):
    raise NotImplementedError


@router.post("/{thread_id}/runs/{run_id}/submit_tool_outputs", tags=['openai'],response_model=RunObject)
async def submit_tool_outputs_to_run(thread_id: str, run_id: str, submit: RunSubmit):
    raise NotImplementedError


@router.post("/{thread_id}/runs/{run_id}/cancel",tags=['openai'], response_model=RunObject)
async def cancel_run(thread_id: str, run_id: str):
    ctx: ThreadContext = await get_thread_context_manager().get_context_by_thread_id(thread_id)
    if ctx is not None:
        if ctx.run is None:
            logger.warn(f'Run {ctx.run.id} is expected to be in_progress, but no context is found')
            raise internal_server_error('ctx do not have run')
        
        if ctx.run.id == run_id:
            logger.info(f'Cancelling thread: {thread_id} and run: {run_id}')
            ctx.run.stream_response_with_event(RunObject.Status.cancelling)
            return ctx.run
        else:
            run = runs_manager.db_get_run(run_id)
            logger.info(f'Run {run_id} not in this thread context')
            return run 
    else:
        run = runs_manager.db_get_run(run_id)
        logger.info(f'Run {run_id} not in context manager')
        return run 
