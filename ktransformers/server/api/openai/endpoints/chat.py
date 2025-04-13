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

from ktransformers.server.schemas.endpoints.chat import ChatCompletionChunk

# Define own data structure instead of importing from OpenAI
class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Dict[str, Any]] = None
    completion_tokens_details: Optional[Dict[str, Any]] = None

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
            logger.warning(f"Unable to get function，function_name: {function_name}")
    
    logger.info(f"Total {len(extracted_tools)} Functions")
    return extracted_tools

@router.post('/chat/completions', tags=['openai'])
async def chat_completion(request: Request, create: ChatCompletionCreate):
    id = str(uuid4().hex)
    
    # 1. Use system prompts to let models know how to use tools
    enhanced_messages = list(create.messages)
    
    # If there is a tool and the first message is system, add instructions on how to use the tool in the system tip
    if create.tools and len(create.tools) > 0 and (enhanced_messages[0].role == Role.system or enhanced_messages[0].role == Role.user):
        tool_instructions = "你可以使用function_call，函数调用功能，目前，你可以使用以下工具\n\n"
        for tool in create.tools:
            tool_instructions += f" \"function\":{{\"name\" : {tool.function.name},\"description\" : {tool.function.description} , \"parameters\" : {tool.function.parameters}}}\n"
        
        # Modify tool usage guidelines to encourage JSON output
        tool_instructions += "name为函数名称，description为函数功能的描述，parameters中含有函数需要使用的参数和参数的描述, 其中required为必要参数\n"
        tool_instructions += "工具仅在用户明确提出，或者你认为需要调用工具的时候调用，注意，当需要高度实时性的信息比如时间或者最近的事情等，优先调用工具来获取！。当确实调用工具的关键信息时，你可以先向用户索取关键信息再调用工具\n"
        tool_instructions += "\n当你需要使用工具时，请以下列格式输出，格式为：\n"
        tool_instructions += '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>name\n```json {"参数名": "参数值","参数名2": "参数值2"...}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n'
        tool_instructions += '示例: \n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>the_functnion_name_will_be_called\n```json {"arg1": "value1","arg2": "value2"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n'
        tool_instructions += "这样可以调用名为\"the_functnion_name_will_be_called\",并将value1和value2传入参数arg1,arg2\n"
        tool_instructions += "不要尝试解释你在做什么，直接输出工具函数调用即可。确保函数调用语句格式正确且完整。"
        
        enhanced_messages[0].content = enhanced_messages[0].content + "\n\n" + tool_instructions
    
    # Requests processed
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
            
            # Collect the full output of the model, but specialize in processing tool calls
            full_content = ""
            buffer = ""  # Used to temporarily store the current block of text
            tool_call_mode = False  # Mark if a tool call is being processed
            tool_calls = []  # Store all detected tool calls
            
            # Customize model special tokens
            tool_calls_begin_marker = "<｜tool▁calls▁begin｜>"
            tool_call_begin_marker = "<｜tool▁call▁begin｜>"
            tool_sep_marker = "<｜tool▁sep｜>"
            tool_call_end_marker = "<｜tool▁call▁end｜>"
            tool_calls_end_marker = "<｜tool▁calls▁end｜>"
            
            async for res in interface.inference(input_message, id, create.temperature, create.top_p):
                if isinstance(res, RawUsage):
                    # Final return on utilization
                    raw_usage = res
                    chunk.choices = []
                    chunk.usage = CompletionUsage(
                        prompt_tokens=raw_usage.prefill_count,
                        completion_tokens=raw_usage.decode_count,
                        total_tokens=raw_usage.prefill_count + raw_usage.decode_count
                    )
                    yield chunk
                elif isinstance(res, tuple) and len(res) == 2:
                    token, finish_reason = res
                    
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
                                # Parsing Calling Text Extraction Tool Calling Information
                                
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
        
        async for res in interface.inference(input_message, id, create.temperature, create.top_p):
            if isinstance(res, RawUsage):
                raw_usage = res
                usage = CompletionUsage(
                    prompt_tokens=raw_usage.prefill_count,
                    completion_tokens=raw_usage.decode_count,
                    total_tokens=raw_usage.prefill_count + raw_usage.decode_count
                )
            elif isinstance(res, tuple) and len(res) == 2:
                token, finish_reason = res
                
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
                        try:
                            # Parsing Calling Text Extraction Tool Calling Information
                            full_tool_call = buffer
                            
                            # Extract function name
                            function_name_start = full_tool_call.find(tool_sep_marker) + len(tool_sep_marker)
                            function_name_end = full_tool_call.find("\n", function_name_start)
                            function_name = full_tool_call[function_name_start:function_name_end].strip()
                            
                            # Extract JSON Parameters - Extracts the content between ```json and ```.
                            json_pattern = r'```json\s*(.*?)\s*```'
                            json_match = re.search(json_pattern, full_tool_call, re.DOTALL)
                            
                            if json_match:
                                arguments_str = json_match.group(1).strip()
                                # Generate tool call IDs
                                tool_call_id = f"call_{uuid4().hex[:24]}"
                                
                                # Add to tool call list
                                tool_calls.append({
                                    "id": tool_call_id,
                                    "index": 0,
                                    "type": "function",
                                    "function": {
                                        "name": function_name,
                                        "arguments": arguments_str
                                    }
                                })
                                
                                # If the tool call is successfully parsed, set the reason for completion
                                finish_reason = "tool_calls"
                                
                                # reset state
                                tool_call_mode = False
                                buffer = ""
                            else:
                                # JSON extraction failed, probably incomplete formatting
                                logger.warning("Failed to extract JSON from tool call")
                                tool_call_mode = False
                                buffer = ""
                        except Exception as e:
                            logger.error(f"Error processing tool call: {e}")
                            tool_call_mode = False
                            buffer = ""
                            
        # Build Response
        response = {
            "id": id,
            "object": "chat.completion",
            "created": int(time()),
            "model": Config().model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None if tool_calls else full_content,
                    "tool_calls": tool_calls if tool_calls else None
                },
                "finish_reason": finish_reason or "stop"
            }],
            "usage": usage.__dict__,
            "system_fingerprint": f"fp_{uuid4().hex[:12]}"
        }
        
        return response
