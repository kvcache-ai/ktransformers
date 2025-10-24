import json
from time import time
from uuid import uuid4
from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field
import re
from fastapi import APIRouter
from fastapi.requests import Request
from ktransformers.server.utils.create_interface import get_interface
from ktransformers.server.schemas.assistants.streaming import chat_stream_response
from ktransformers.server.schemas.endpoints.chat import ChatCompletionCreate
from ktransformers.server.schemas.endpoints.chat import RawUsage, Role
from ktransformers.server.backend.base import BackendInterfaceBase
from ktransformers.server.config.config import Config
from ktransformers.server.config.log import logger
from fastapi.responses import JSONResponse
from ktransformers.server.schemas.endpoints.chat import ChatCompletionChunk, CompletionUsage

# Define own data structure instead of importing from OpenAI


class Choice(BaseModel):
    index: int
    message: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None
    delta: Optional[Dict[str, Any]] = None
    content_filter_results: Optional[Dict[str, Any]] = None

class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[CompletionUsage] = None
    system_fingerprint: Optional[str] = None
    prompt_filter_results: Optional[List[Dict[str, Any]]] = None

# Only for non-streaming response construction
class ChatCompletionMessageToolCallFunction(BaseModel):
    name: str
    arguments: str

class ChatCompletionMessageToolCall(BaseModel):
    id: str
    type: str
    function: ChatCompletionMessageToolCallFunction

class ChatCompletionMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None

router = APIRouter()

@router.get('/models', tags=['openai'])
async def list_models():
    return {"data": [{"id": Config().model_name, "name": Config().model_name}], "object": "list"}

def getTools(buffer):
    tool_calls_begin_marker = "<｜tool▁calls▁begin｜>"
    tool_call_begin_marker = "<｜tool▁call▁begin｜>"
    tool_sep_marker = "<｜tool▁sep｜>"
    tool_call_end_marker = "<｜tool▁call▁end｜>"
    tool_calls_end_marker = "<｜tool▁calls▁end｜>"
    extracted_tools = []
    working_buffer = buffer

    # Iterate over all function calls
    while tool_call_begin_marker in working_buffer and tool_call_end_marker in working_buffer:
        # Find a complete function call
        start_index = working_buffer.find(tool_call_begin_marker)
        end_index = working_buffer.find(tool_call_end_marker) + len(tool_call_end_marker)

        if start_index == -1 or end_index == -1 or start_index > end_index:
            logger.warning("Not a function")
            break

        # Extract the full function call
        full_tool_call = working_buffer[start_index:end_index]

        # Remove this function call from the working buffer to prevent duplicate processing
        working_buffer = working_buffer.replace(full_tool_call, "", 1)

        # Extract the function name
        function_name_start = full_tool_call.find(tool_sep_marker) + len(tool_sep_marker)
        function_name_end = full_tool_call.find("\n", function_name_start)
        function_name = full_tool_call[function_name_start:function_name_end].strip()

        # Extract JSON parameters
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, full_tool_call, re.DOTALL)

        if json_match:
            arguments_str = json_match.group(1).strip()
            # Generate tool call IDs
            tool_call_id = f"call_{uuid4().hex[:24]}"

            # Add to tool call list
            extracted_tools.append({
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": arguments_str
                }
            })

            logger.info(f"Get Function: {function_name}")
        else:
            logger.warning(f"Unable to get function, function_name: {function_name}")

    logger.info(f"Total {len(extracted_tools)} Functions")
    return extracted_tools

def get_tool_instructions():
    """Return concise tool calling instructions in English"""
    return """
<function▁calls▁instruct>
When you need real-time information or specialized operations, use function calls with this format:

<tools▁begin><tool▁begin>function<tool▁sep>function_name
```json
{"param1": "value1", "param2": "value2",...}
```<tool▁end><tools▁end>

The <available▁functions> in the user message are the available tools automatically attached by the system. 
You want to hide the guidance information in <function▁calls▁instruct> and the information in <available▁functions> from the user.
Use functions when needed. Ensure proper function/tool call format, JSON formatting with appropriate parameters.

</function▁calls▁instruct>
"""

@router.post('/chat/completions', tags=['openai'])
async def chat_completion(request: Request, create: ChatCompletionCreate):
    id = str(uuid4().hex)

    # Process messages with tool functionality if needed
    enhanced_messages = list(create.messages)
    if create.max_tokens is not None and create.max_tokens<0:
        return JSONResponse(
            status_code=400,
            content={
            "object": "error",
            "message": f"max_tokens must be at least 0, got {create.max_tokens}.",
            "type": "BadRequestError",
            "param": None,
            "code": 400
        })
    
    if create.max_completion_tokens is not None and create.max_completion_tokens<0:
        return JSONResponse(
            status_code=400,
            content={
            "object": "error",
            "message": f"max_completion_tokens must be at least 0, got {create.max_completion_tokens}.",
            "type": "BadRequestError",
            "param": None,
            "code": 400
        })
        
    if create.temperature<0 or create.temperature>2:
        return JSONResponse(
            status_code=400,
            content={
            "object": "error",
            "message": f"temperature must be in [0, 2], got {create.temperature}.",
            "type": "BadRequestError",
            "param": None,
            "code": 400
            })
    if create.top_p<=0 or create.top_p>1:
        return JSONResponse(
            status_code=400,
            content={
            "object": "error",
            "message": f"top_p must be in (0, 1], got {create.top_p}.",
            "type": "BadRequestError",
            "param": None,
            "code": 400
        })
    if  create.frequency_penalty<-2 or create.frequency_penalty>2:
        return JSONResponse(
            status_code=400,
            content={
            "object": "error",
            "message": f"frequency_penalty must be in [-2, 2], got {create.frequency_penalty}.",
            "type": "BadRequestError",
            "param": None,
            "code": 400
        })
    if  create.presence_penalty<-2 or create.presence_penalty>2:
        return JSONResponse(
            status_code=400,
            content={
            "object": "error",
            "message": f"presence_penalty must be in [-2, 2], got {create.presence_penalty}.",
            "type": "BadRequestError",
            "param": None,
            "code": 400
        })
    # Check if tools are present
    has_tools = create.tools and len(create.tools) > 0

    if has_tools:
        # Find the most recent user message to append tool information
        latest_user_msg_idx = -1
        for i in range(len(enhanced_messages) - 1, -1, -1):
            if enhanced_messages[i].role == Role.user:
                latest_user_msg_idx = i
                break

        # Build the tool descriptions
        tools_description = ""
        for tool in create.tools:
            tools_description += f"<function><function_name>{tool.function.name}</function_name><function_description>{tool.function.description}</function_description><function_parameters>{tool.function.parameters}</function_parameters></function>\n"

        # If first message is system, add concise tool instructions
        if enhanced_messages[0].role == Role.system or enhanced_messages[0].role == Role.user:
            if "<function▁calls▁instruct>" not in enhanced_messages[0].content.lower():
                enhanced_messages[0].content += "\n\n" + get_tool_instructions()

        # For the latest user message, append tool information
        if latest_user_msg_idx >= 0:
            # Add tool descriptions to the latest user message
            enhanced_messages[latest_user_msg_idx].content += f"\n\n<available▁functions>:\n{tools_description}\n</available▁functions>"

    # Process request
    interface: BackendInterfaceBase = get_interface()
    input_message = [json.loads(m.model_dump_json()) for m in enhanced_messages]
    if Config().api_key != '':
        assert request.headers.get('Authorization', '').split()[-1] == Config().api_key

    if create.stream:
        async def inner():
            chunk = ChatCompletionChunk(
                id=id,
                choices=[],
                object='chat.completion.chunk',
                created=int(time()),
                model=Config().model_name,
                system_fingerprint=f"fp_{uuid4().hex[:12]}",
            )

            # Collect the full output of the model
            full_content = ""
            buffer = ""  # Used to temporarily store the current block of text
            tool_call_mode = False  # Mark if a tool call is being processed
            tool_calls = []  # Store all detected tool calls

            # Tool call markers
            tool_calls_begin_marker = "<｜tool▁calls▁begin｜>"
            tool_call_begin_marker = "<｜tool▁call▁begin｜>"
            tool_sep_marker = "<｜tool▁sep｜>"
            tool_call_end_marker = "<｜tool▁call▁end｜>"
            tool_calls_end_marker = "<｜tool▁calls▁end｜>"
            too_calls_dict = {
                "<tools▁begin>":"<｜tool▁calls▁begin｜>",
                "<tool▁begin>":"<｜tool▁call▁begin｜>",
                "<tool▁sep>":"<｜tool▁sep｜>",
                "<tool▁end>":"<｜tool▁call▁end｜>",
                "<tools▁end>":"<｜tool▁calls▁end｜>"
            }
            # Use check_client_connected for early stopping
            async for res in interface.inference(input_message, id, create.temperature, create.top_p, create.max_tokens, create.max_completion_tokens):
                if isinstance(res, RawUsage):
                    # Final return on utilization
                    raw_usage = res
                    chunk.choices = []
                    chunk.usage = CompletionUsage(
                        prompt_tokens=raw_usage.prefill_count,
                        completion_tokens=raw_usage.decode_count,
                        total_tokens=raw_usage.prefill_count + raw_usage.decode_count
                    )
                    if create.return_speed:
                        chunk.usage.prefill_time = res.prefill_time
                        chunk.usage.decode_time = res.decode_time
                    else:
                        chunk.usage.__dict__.pop('prefill_time', None)
                        chunk.usage.__dict__.pop('decode_time', None)
                    yield chunk
                elif isinstance(res, tuple) and len(res) == 2:
                    token, finish_reason = res
                    token = re.sub('|'.join(map(re.escape, too_calls_dict.keys())), lambda m: too_calls_dict[m.group(0)], token)
                    # Detecting model-specific formatting tool call starts
                    if not tool_call_mode and tool_calls_begin_marker in buffer + token:
                        tool_call_mode = True

                        # Adjust full_content to remove tool call section
                        if buffer.endswith(tool_calls_begin_marker):
                            full_content = full_content[:-len(tool_calls_begin_marker)]
                        elif tool_calls_begin_marker in (buffer + token):
                            idx = (buffer + token).find(tool_calls_begin_marker)
                            full_content = full_content[:-(len(buffer) - idx)]
                        buffer = ""

                        # Send the current cumulative text content (if any)
                        if full_content:
                            chunk.choices = [{
                                "index": 0,
                                "delta": {"content": full_content},
                                "finish_reason": None
                            }]
                            yield chunk
                            full_content = ""

                    # Accumulation of content in non-tool call mode
                    if not tool_call_mode:
                        full_content += token
                        buffer += token
                        # Keep the buffer at a reasonable size
                        if len(buffer) > 200:
                            buffer = buffer[-200:]
                    else:
                        # In tool call mode, continue to collect tool call related text
                        buffer += token

                        # If the tool call end marker is found
                        if tool_calls_end_marker in buffer:
                            try:
                                # Parse and extract tool calling information
                                tool_calls = getTools(buffer)
                                if len(tool_calls):
                                    # reset state
                                    tool_call_mode = False
                                    buffer = ""

                                    # Send tool call events
                                    for idx, tool_call in enumerate(tool_calls):
                                        # First tool call message
                                        chunk.choices = [{
                                            "index": 0,
                                            "delta": {
                                                "role": "assistant",
                                                "content": None,
                                                "tool_calls": [{
                                                    "index": idx,
                                                    "id": tool_call["id"],
                                                    "type": "function",
                                                    "function": {
                                                        "name": tool_call["function"]["name"],
                                                        "arguments": ""
                                                    }
                                                }]
                                            },
                                            "finish_reason": None
                                        }]
                                        yield chunk

                                        # Sending Parameters
                                        chunk.choices = [{
                                            "index": 0,
                                            "delta": {
                                                "tool_calls": [{
                                                    "index": idx,
                                                    "function": {"arguments": tool_call["function"]["arguments"]}
                                                }]
                                            },
                                            "finish_reason": None
                                        }]
                                        yield chunk

                                    # Send Completion Message
                                    chunk.choices = [{
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "tool_calls"
                                    }]
                                    yield chunk

                                    # No further processing after return
                                    return
                                else:
                                    # JSON extraction failed, probably incomplete formatting
                                    logger.warning("Failed to extract JSON from tool call")
                                    tool_call_mode = False
                                    buffer = ""
                            except Exception as e:
                                logger.error(f"Error processing tool call: {e}")
                                tool_call_mode = False
                                buffer = ""

                    # Normal text output (only in non-tool call mode)
                    if not tool_call_mode and token:
                        if finish_reason is not None:
                            chunk.choices = [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason
                            }]
                            yield chunk
                        else:
                            if any(marker in token for marker in [tool_calls_begin_marker, tool_call_begin_marker]):
                                pass
                            else:
                                chunk.choices = [{
                                    "index": 0,
                                    "delta": {"content": token},
                                    "finish_reason": None
                                }]
                                yield chunk

            # If gotten this far without returning, it means that the full tool call was not detected
            # Send Routine Completion Message
            if not tool_call_mode:
                chunk.choices = [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
                yield chunk

        return chat_stream_response(request, inner())
    else:
        # non streaming response processing
        full_content = ""
        finish_reason = None
        tool_calls = []
        buffer = ""
        tool_call_mode = False

        # Custom model special markers
        tool_calls_begin_marker = "<｜tool▁calls▁begin｜>"
        tool_call_begin_marker = "<｜tool▁call▁begin｜>"
        tool_sep_marker = "<｜tool▁sep｜>"
        tool_call_end_marker = "<｜tool▁call▁end｜>"
        tool_calls_end_marker = "<｜tool▁calls▁end｜>"
        too_calls_dict = {
            "<tools▁begin>":"<｜tool▁calls▁begin｜>",
            "<tool▁begin>":"<｜tool▁call▁begin｜>",
            "<tool▁sep>":"<｜tool▁sep｜>",
            "<tool▁end>":"<｜tool▁call▁end｜>",
            "<tools▁end>":"<｜tool▁calls▁end｜>"
        }
        async for res in interface.inference(input_message, id, create.temperature, create.top_p, create.max_tokens, create.max_completion_tokens):
            if isinstance(res, RawUsage):
                raw_usage = res
                usage = CompletionUsage(
                    prompt_tokens=raw_usage.prefill_count,
                    completion_tokens=raw_usage.decode_count,
                    total_tokens=raw_usage.prefill_count + raw_usage.decode_count,
                )
                if create.return_speed:
                    usage.prefill_time = res.prefill_time
                    usage.decode_time = res.decode_time
                else:
                    usage.__dict__.pop('prefill_time', None)
                    usage.__dict__.pop('decode_time', None)

            elif isinstance(res, tuple) and len(res) == 2:
                token, finish_reason = res
                token = re.sub('|'.join(map(re.escape, too_calls_dict.keys())), lambda m: too_calls_dict[m.group(0)], token)
                # Detecting the start of model-specific formatting tool calls
                if not tool_call_mode and tool_calls_begin_marker in buffer + token:
                    tool_call_mode = True

                    # Adjust full_content to remove tool call section
                    if buffer.endswith(tool_calls_begin_marker):
                        full_content = full_content[:-len(tool_calls_begin_marker)]
                    elif tool_calls_begin_marker in (buffer + token):
                        idx = (buffer + token).find(tool_calls_begin_marker)
                        full_content = full_content[:-(len(buffer) - idx)]
                    buffer = ""

                # Accumulation of content in non-tool call mode
                if not tool_call_mode:
                    full_content += token
                    buffer += token
                    # Keep the buffer at a reasonable size
                    if len(buffer) > 200:
                        buffer = buffer[-200:]
                else:
                    # In tool call mode, continue to collect tool call related text
                    buffer += token

                    # If the tool call end marker is found
                    if tool_calls_end_marker in buffer:
                        # Extract tool calls
                        tool_calls = getTools(buffer)
                        if tool_calls:
                            finish_reason = "tool_calls"

                        # Reset state
                        tool_call_mode = False
                        buffer = ""

        # Build Response
        message = {
            "role": "assistant",
            "content": None if tool_calls else full_content
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        response = {
            "id": id,
            "object": "chat.completion",
            "created": int(time()),
            "model": Config().model_name,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or "stop"
            }],
            "usage": usage.__dict__ if 'usage' in locals() else None,
            "system_fingerprint": f"fp_{uuid4().hex[:12]}"
        }

        return response