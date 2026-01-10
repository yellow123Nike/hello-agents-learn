import asyncio
import copy
import json
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional, Iterator
import uuid
from openai import APIConnectionError, AsyncOpenAI, OpenAI, RateLimitError, Timeout
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
import tiktoken

from hello_agents.core.exceptions import HelloAgentsException
from hello_agents.core.llm.llm_prompt import SENSITIVE_PATTERNS, STRUCT_PARSE_TOOL_SYSTEM_PROMPT
from hello_agents.core.agent.agent_schema import AgentContext
from hello_agents.core.llm.llm_schema import PROVIDERS, FunctionCallType, LLMParams, ToolCallResponse, ToolChoice
from hello_agents.core.llm.message import Message
from hello_agents.core.llm.string_util import text_desensitization
from hello_agents.core.tool.tool_collection import ToolCollection
from hello_agents.core.tool.tool_schema import ToolCall


class LLMClient:
    def __init__(self, params: LLMParams, provider_name: str):
        self.params = params
        self.provider = PROVIDERS[provider_name]
        self.client = AsyncOpenAI(
            api_key=params.api_key,
            base_url=params.base_url,
        )
        self.function_call_type = None

    def build_request_params(self, messages, stream, tools=None, tool_choice=None):
        payload = {
            "model": self.params.model_name,
            "messages": messages,
            "temperature": self.params.temperature,
            "stream": stream,
        }

        payload[self.provider.max_tokens_field] = self.params.max_tokens
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        return payload

    # 格式化消息

    def format_messages(
        self,
        messages: List[Message]
    ):
        formatted = []
        for msg in messages:
            message_map: Dict[str, Any] = {}
            # ===== multimodal =====
            # 1.处理 base64 图像
            if msg.base64_image:
                multimodal = []
                multimodal.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{msg.base64_image}"
                    }
                })
                multimodal.append({
                    "type": "text",
                    "text": msg.content
                })

                message_map["role"] = msg.role.value
                message_map["content"] = multimodal

            # ===== tool calls =====
                # Claude 把「工具调用」当作一种消息类型（content block），
                # GPT 把「工具调用」当作 message 的一个字段（tool_calls）
            elif msg.tool_calls:
                message_map["role"] = msg.role.value
                if self.provider.tool_call_style == "claude":
                    claude_calls = []
                    for tc in msg.tool_calls:
                        claude_calls.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": json.loads(tc.function.arguments),
                        })
                    message_map["content"] = claude_calls
                else:
                    message_map["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in msg.tool_calls
                    ]

            # ===== tool result =====
                # 工具执行结果重新喂给大模型
            elif msg.tool_call_id:
                # 执行结果脱敏
                content = text_desensitization(
                    msg.content,
                    SENSITIVE_PATTERNS,
                )

                if self.provider.tool_call_style == "claude":
                    message_map["role"] = "user"
                    message_map["content"] = [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": content,
                    }]
                else:
                    message_map["role"] = msg.role.value
                    message_map["content"] = content
                    message_map["tool_call_id"] = msg.tool_call_id

            # ===== normal text =====
            else:
                message_map["role"] = msg.role.value
                message_map["content"] = msg.content

            formatted.append(message_map)

        return formatted

    def count_message_tokens(self, message: Dict[str, Any]) -> int:
        """
        统计单条 message 的 token 数
        """
        try:
            encoding = tiktoken.encoding_for_model(self.params.model_name)
        except KeyError:
            # fallback（非常重要，避免模型名不识别）
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens = 0
        # role tokens（OpenAI 固定开销）
        tokens += 4  # 每条 message 的结构开销
        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    tokens += len(encoding.encode(item.get("text", "")))
                elif item.get("type") == "image_url":
                    tokens += 85
        else:
            tokens += len(encoding.encode(str(content)))

        return tokens

    # token截断:倒序贪心+user边界对齐
    def truncate_message(
        self,
        context: AgentContext,
        messages: List[Dict[str, Any]],
        max_input_tokens: int
    ):
        if not messages or max_input_tokens < 0:
            return messages

        truncated_messages: List[Dict[str, Any]] = []
        remaining_tokens = max_input_tokens

        system = messages[0]
        if system.get("role") == "system":
            remaining_tokens -= self.count_message_tokens(system)

        # 从后往前取
        for message in reversed(messages):
            message_tokens = self.count_message_tokens(message)
            if remaining_tokens >= message_tokens:
                truncated_messages.insert(0, message)
                remaining_tokens -= message_tokens
            else:
                break

        # 保证第一条非 system 消息是 user
        while truncated_messages:
            first = truncated_messages[0]
            if first.get("role") != "user":
                truncated_messages.pop(0)
            else:
                break

        if system.get("role") == "system":
            truncated_messages.insert(0, system)

        return truncated_messages

    def _prepare_messages(
        self,
        context: AgentContext,
        messages: List[Message],
        system_msgs: Optional[Message],
    ) -> List[dict]:
        # -------- 1.1 格式化 messages --------
        if system_msgs:
            formatted_system_msgs = self.format_messages(
                [system_msgs]
            )
            formatted_messages = list(formatted_system_msgs)
            formatted_messages.extend(
                self.format_messages(messages)
            )
        else:
            formatted_messages = self.format_messages(
                messages
            )
        # -------- 1.2 截断输入 --------
        if self.params.max_tokens is not None:
            formatted_messages = self.truncate_message(
                context=context,
                messages=formatted_messages,
                max_input_tokens=self.params.max_tokens,
            )

        return formatted_messages

    # function_call-param
    def add_function_name_param(
        self,
        parameters: Dict[str, Any],
        tool_name: str,
    ):
        """
        """
        new_parameters = copy.deepcopy(parameters)
        new_required = ["function_name"]
        if "required" in parameters and parameters["required"] is not None:
            new_required.extend(parameters["required"])
        new_parameters["required"] = new_required
        new_properties: Dict[str, Any] = {}

        function_name_map = {
            "description": f"默认值为工具名: {tool_name}",
            "type": "string",
        }
        new_properties["function_name"] = function_name_map

        if "properties" in parameters and parameters["properties"] is not None:
            new_properties.update(parameters["properties"])

        new_parameters["properties"] = new_properties

        return new_parameters

    def to_openai_tool_choice(
        self,
        tool_choice: ToolChoice,
        forced_tool_name: Optional[str] = None,
    ):
        """
        将内部 ToolChoice 映射为 OpenAI chat.completions 的 tool_choice 参数。
        - NONE/AUTO: 直接返回字符串
        - REQUIRED:
            - 如果指定 forced_tool_name：返回强制调用某工具的 object
            - 否则：降级为 "auto"（避免 OpenAI 不支持 "required" 导致报错）
        """
        if tool_choice == ToolChoice.NONE:
            return "none"
        if tool_choice == ToolChoice.AUTO:
            return "auto"

        # REQUIRED
        if forced_tool_name:
            return {"type": "function", "function": {"name": forced_tool_name}}

        # 无法明确强制哪一个工具时，不建议传 "required"
        # 因为 chat.completions 未必接受；降级为 auto 更稳
        return "auto"

    async def ask_llm_once(
        self,
        context: AgentContext,
        messages: List[Message],
        system_msgs: Optional[List[Message]] = None,
    ) -> str:
        try:
            formatted_messages = self._prepare_messages(
                context, messages, system_msgs
            )

            params = {"messages": formatted_messages, "stream": False}

            response = await self.call_openai(params)

            if (
                not response
                or not response.choices
                or response.choices[0].message.content is None
            ):
                raise ValueError("Empty or invalid response from LLM")

            return response.choices[0].message.content

        except Exception:
            self.age("%s ask_llm_once failed", context.request_id)
            raise

    async def ask_llm_stream(
        self,
        context: AgentContext,
        messages: List[Message],
        system_msgs: Optional[List[Message]] = None,
    ):
        try:
            formatted_messages = self._prepare_messages(
                context, messages, system_msgs
            )

            params = {"messages": formatted_messages, "stream": True}

            async for chunk in self.call_openai_stream(params):
                yield chunk

        except Exception:
            context.printer.send(
                "%s ask_llm_stream failed", context.request_id)
            raise

    def _normalize_enum(self, value, enum_cls, name: str):
        if isinstance(value, enum_cls):
            return value
        if isinstance(value, str):
            try:
                return enum_cls(value)
            except ValueError:
                raise ValueError(
                    f"Invalid {name}: {value}, "
                    f"must be one of {[e.value for e in enum_cls]}"
                )
        raise TypeError(
            f"{name} must be {enum_cls.__name__} or str, got {type(value)}"
        )

    async def ask_tool(
        self,
        context: AgentContext,
        messages: List[Message],
        tools: ToolCollection,
        tool_choice: ToolChoice | str,
        system_msgs: Optional[Message],
        function_call_type: FunctionCallType | str = FunctionCallType.FUNCTION_CALL
    ) -> ToolCallResponse:
        try:
            # ===== 1. ToolChoice 校验=====
            self.function_call_type = function_call_type
            tool_choice = self._normalize_enum(
                tool_choice, ToolChoice, "tool_choice")
            function_call_type = self._normalize_enum(
                function_call_type,
                FunctionCallType,
                "function_call_type"
            )
            start_time = time.time()
            # ===== 2. 构造 OpenAI tools（对齐 function_call 分支） =====
            formatted_tools: list[dict] = []
            string_builder: list[str] = []
            if function_call_type is FunctionCallType.STRUCT_PARSE:
                # ===== struct_parse 分支 =====
                string_builder.append(STRUCT_PARSE_TOOL_SYSTEM_PROMPT)
                # ---------- base tool ----------
                for tool in tools.tool_map.values():
                    function_map = {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": self.add_function_name_param(
                            tool.to_params(),
                            tool.name,
                        ),
                    }
                    string_builder.append(
                        f"- `{tool.name}`\n```json {json.dumps(function_map, ensure_ascii=False)} ```\n"
                    )

                # ---------- mcp tool ----------
                for tool in tools.mcp_tool_map.values():
                    parameters = json.loads(tool.parameters)
                    function_map = {
                        "name": tool.name,
                        "description": tool.desc,
                        "parameters": self.add_function_name_param(
                            parameters,
                            tool.name,
                        ),
                    }
                    string_builder.append(
                        f"- `{tool.name}`\n```json {json.dumps(function_map, ensure_ascii=False)} ```\n"
                    )
                struct_prompt = "\n".join(string_builder)
                system_msgs.content = (
                    (system_msgs.content or "")
                    + "\n"
                    + struct_prompt
                )
            else:
                # ========= base tool =========
                for tool in tools.tool_map.values():
                    function_map = {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.to_params(),  # 注意：没有 add_function_name_param
                    }

                    tool_map = {
                        "type": "function",
                        "function": function_map,
                    }

                    formatted_tools.append(tool_map)

                # ========= mcp tool =========
                for tool in tools.mcp_tool_map.values():
                    parameters = json.loads(tool.parameters)

                    function_map = {
                        "name": tool.name,
                        "description": tool.desc,
                        "parameters": parameters,
                    }

                    tool_map = {
                        "type": "function",
                        "function": function_map,
                    }

                    formatted_tools.append(tool_map)

            # ===== 3. 格式化消息 =====
            formatted_messages = self._prepare_messages(
                context, messages, system_msgs
            )

            # ===== 4. 调用 OpenAI =====
            payload = self.build_request_params(
                messages=formatted_messages, stream=False, tools=formatted_tools,
                tool_choice=self.to_openai_tool_choice(tool_choice)
            )
            response = await asyncio.wait_for(self.client.chat.completions.create(
                **payload
            ),
                timeout=240,
            )

            # ===== 5. 解析响应 =====
            if not response.choices or response.choices[0].message is None:
                raise ValueError("Invalid or empty response from LLM")

            choice = response.choices[0]
            message = choice.message

            content = message.content if message.content != "null" else None
            tool_calls: List["ToolCall"] = []
            if function_call_type is FunctionCallType.STRUCT_PARSE:
                pattern = r"```json\s*([\s\S]*?)\s*```"
                content = re.findall(pattern, content or "")
                for json_block in content:
                    try:
                        data = json.loads(json_block)
                        tool_name = data.pop("function_name", None)
                        if not tool_name:
                            continue

                        tool_calls.append(
                            ToolCall(
                                id=str(uuid.uuid4()),
                                type="function",
                                function=ToolCall.Function(
                                    name=tool_name,
                                    arguments=json.dumps(
                                        data, ensure_ascii=False),
                                ),
                            )
                        )
                    except Exception:
                        # 对齐 Java：解析失败直接忽略
                        continue
            else:
                if message.tool_calls:
                    for tc in message.tool_calls:
                        tool_calls.append(
                            ToolCall(
                                id=tc.id,
                                type=tc.type,
                                function=ToolCall.Function(
                                    name=tc.function.name,
                                    arguments=tc.function.arguments,
                                ),
                            )
                        )
            finish_reason = choice.finish_reason
            # ===== usage =====
            total_tokens = response.usage.total_tokens if response.usage else None
            # ===== duration =====
            duration_ms = int((time.time() - start_time) * 1000)
            return ToolCallResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                total_tokens=total_tokens,
                duration=duration_ms,
            )

        except Exception as e:
            print(f"%s Unexpected error in ask_tool: %s",
                  context.request_id,
                  str(e),
                  )
            raise

    @retry(
        stop=stop_after_attempt(3),                 # 最多重试 3 次
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避
        retry=retry_if_exception_type(
            (
                RateLimitError,
                APIConnectionError,
                Timeout,
                asyncio.TimeoutError,
            )
        ),
        reraise=True,   # 最终失败时抛出原异常
    )
    async def call_openai(
        self,
        params
    ):
        payload = self.build_request_params(
            messages=params["messages"], stream=params["stream"],
        )
        response = await asyncio.wait_for(self.client.chat.completions.create(
            **payload
        ), timeout=240)

        return response

    @retry(
        stop=stop_after_attempt(3),                 # 最多重试 3 次
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避
        retry=retry_if_exception_type(
            (
                RateLimitError,
                APIConnectionError,
                Timeout,
                asyncio.TimeoutError,
            )
        ),
        reraise=True,   # 最终失败时抛出原异常
    )
    async def call_openai_stream(
        self,
        params
    ):
        payload = self.build_request_params(
            messages=params["messages"], stream=params["stream"],
        )
        response = await asyncio.wait_for(self.client.chat.completions.create(
            **payload
        ), timeout=240)

        async for event in response:
            choice = event.choices[0]
            delta = choice.delta

            if delta and delta.content:
                yield delta.content
