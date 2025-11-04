import asyncio
from typing import AsyncIterable, List, Union

from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ktransformers.server.schemas.assistants.runs import RunStreamResponse
from ktransformers.server.schemas.endpoints.chat import ChatCompletionChunk
from ktransformers.server.config.log import logger
from ktransformers.server.schemas.base import Object
from ktransformers.server.schemas.assistants.messages import ContentType, ImageFileObject, ImageUrlObject, MessageObject, Text, TextObject


class TextObjectWithIndex(TextObject):
    index: int


class ImageFileObjectWithIndex(ImageFileObject):
    index: int


class ImageUrlObjectWithIndex(ImageUrlObject):
    index: int


ContentWithIndex = Union[TextObjectWithIndex,
                         ImageFileObjectWithIndex, ImageUrlObjectWithIndex]


class MessageDeltaImpl(BaseModel):
    # role: Optional[str]
    content: List[ContentWithIndex]


class MessageDelta(Object):
    delta: MessageDeltaImpl

    def to_stream_reply(self):
        return f"event: thread.message.delta\ndata: {self.model_dump_json()}\n\n"


def text_delta(index: int, text: str):
    return MessageDeltaImpl(content=[TextObjectWithIndex(index=index, type=ContentType.text, text=Text(value=text))])


def append_message_delta(self: MessageObject, text: str):

    if len(self.content) == 0:
        self.content.append(TextObject(type=ContentType.text,
                            text=Text(value=''), delta_index=0))

    text_object: TextObject = self.content[0]
    if text_object.filter_append(text):
        return MessageDelta(id=self.id, object="thread.message.delta", delta=text_delta(text_object.delta_index, text))
    else:
        return None


MessageObject.append_message_delta = append_message_delta


class RunStepDeltaImpl(BaseModel):
    pass


class RunStepDelta(Object):
    delta: RunStepDeltaImpl

    def to_stream_reply(self):
        return f"event: thread.run.step.delta\ndata: {self.model_dump_json()}\n\n"


class Done():
    def to_stream_reply(self):
        return f"data: [DONE]\n\n"


async def check_client_link(request: Request, async_events: AsyncIterable):
    async for event in async_events:
        if await request.is_disconnected():
            break
        yield event


async def add_done(async_events: AsyncIterable):
    async for event in async_events:
        yield event
    yield Done()


async def to_stream_reply(async_events: AsyncIterable):
    async for event in async_events:
        if isinstance(event, str):
            yield event
        else:
            yield event.to_stream_reply()


async def filter_api_event(async_events: AsyncIterable):
    async for event in async_events:
        if isinstance(event, MessageDelta) or isinstance(event, RunStepDelta) or isinstance(event, RunStreamResponse) or isinstance(event, Done):
            yield event


async def filter_chat_chunk(async_events: AsyncIterable):
    async for event in async_events:
        if isinstance(event, ChatCompletionChunk):
            yield event


async def filter_by_types(async_events: AsyncIterable, types: List):
    async for event in async_events:
        for type in types:
            if isinstance(event, type):
                yield event
                continue


def api_stream_response(request: Request, async_events: AsyncIterable):
    return StreamingResponse(check_client_link(request, to_stream_reply(add_done(filter_api_event(async_events)))), media_type="text/event-stream")


def chat_stream_response(request: Request, async_events: AsyncIterable):
    return StreamingResponse(check_client_link(request, to_stream_reply(add_done(filter_chat_chunk(async_events)))), media_type="text/event-stream")


def stream_response(request: Request, async_events: AsyncIterable):
    return StreamingResponse(check_client_link(request, to_stream_reply(add_done(async_events))), media_type="text/event-stream")


def check_link_response(request: Request, async_events: AsyncIterable):
    return StreamingResponse(check_client_link(request, async_events), media_type="text/event-stream")


def wrap_async_generator_into_queue(async_events: AsyncIterable) -> asyncio.Queue:
    queue = asyncio.Queue()

    async def inner():
        # logger.debug('run inner')
        async for event in async_events:
            # logger.debug(f'put: {event}')
            await queue.put(event)
            await asyncio.sleep(0)
        # logger.debug(f'put: None')
        await queue.put(None)
    asyncio.create_task(inner())
    return queue


async def unwrap_async_queue(queue: asyncio.Queue) -> AsyncIterable:
    while True:
        events = [await queue.get()]
        events.extend([queue.get_nowait() for _ in range(queue.qsize())])

        logger.debug(f'getting {len(events)} events')
        for event in events:
            if event is None:
                break
            yield event


async def unwrap_async_queue_slow(queue: asyncio.Queue) -> AsyncIterable:
    while True:
        event = await queue.get()
        # logger.debug(f'unwrap_async_queue {event}')
        if event is None:
            break
        yield event
