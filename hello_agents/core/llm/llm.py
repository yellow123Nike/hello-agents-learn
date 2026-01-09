import json
import os
from typing import Any, Dict, List, Literal, Optional, Iterator
from openai import AsyncOpenAI, OpenAI
import tiktoken

from hello_agents.core.exceptions import HelloAgentsException
from hello_agents.core.llm.llm_prompt import SENSITIVE_PATTERNS
from hello_agents.core.agent.agent_schema import AgentContext, LLMParams
from hello_agents.core.llm.message import Message
from hello_agents.core.llm.string_util import text_desensitization


class LLMClient:
    def __init__(self, params: LLMParams):
        self.params = params
        self.is_claude = params.is_claude
        self.client = AsyncOpenAI(
            api_key=params.api_key,
            base_url=params.base_url,
        )
        self.function_call_type = None

    # 格式化消息
    def format_messages(
        self,
        messages: List[Message],
        is_claude: bool,
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
                if is_claude:
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

                if is_claude:
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
                [system_msgs],
                is_claude=self.is_claude,
            )
            formatted_messages = list(formatted_system_msgs)
            formatted_messages.extend(
                self.format_messages(messages, is_claude=self.is_claude)
            )
        else:
            formatted_messages = self.format_messages(
                messages,
                is_claude=self.is_claude,
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
            logger.exception("%s ask_llm_once failed", context.request_id)
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
            logger.exception("%s ask_llm_stream failed", context.request_id)
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
            response = await asyncio.wait_for(self.client.chat.completions.create(
                model=self.params.model_name,
                messages=formatted_messages,
                tools=formatted_tools,
                tool_choice=self.to_openai_tool_choice(tool_choice),
                temperature=self.params.temperature,
                max_tokens=self.params.max_tokens,
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
        response = await asyncio.wait_for(self.client.chat.completions.create(
            model=self.params.model_name,
            messages=params["messages"],
            temperature=self.params.temperature,
            max_tokens=self.params.max_tokens,
            stream=params["stream"],
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
        response = await asyncio.wait_for(self.client.chat.completions.create(
            model=self.params.model_name,
            messages=params["messages"],
            temperature=self.params.temperature,
            max_tokens=self.params.max_tokens,
            stream=params["stream"],
        ), timeout=240)

        async for event in response:
            choice = event.choices[0]
            delta = choice.delta

            if delta and delta.content:
                yield delta.content


# 支持的LLM提供商
SUPPORTED_PROVIDERS = Literal[
    "openai",
    "deepseek",
    "qwen",
    "modelscope",
    "kimi",
    "zhipu",
    "ollama",
    "vllm",
    "local",
    "auto",
    "custom",
]


class HelloAgentsLLM:
    """
    为HelloAgents定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。

    设计理念：
    - 参数优先，环境变量兜底
    - 流式响应为默认，提供更好的用户体验
    - 支持多种LLM提供商
    - 统一的调用接口
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[SUPPORTED_PROVIDERS] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        支持自动检测provider或使用统一的LLM_*环境变量配置。

        Args:
            model: 模型名称，如果未提供则从环境变量LLM_MODEL_ID读取
            api_key: API密钥，如果未提供则从环境变量读取
            base_url: 服务地址，如果未提供则从环境变量LLM_BASE_URL读取
            provider: LLM提供商，如果未提供则自动检测
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 超时时间，从环境变量LLM_TIMEOUT读取，默认60秒
        """
        # 优先使用传入参数，如果未提供，则从环境变量加载
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        self.kwargs = kwargs

        # 自动检测provider或使用指定的provider
        requested_provider = (provider or "").lower() if provider else None
        self.provider = provider or self._auto_detect_provider(
            api_key, base_url)

        if requested_provider == "custom":
            self.provider = "custom"
            self.api_key = api_key or os.getenv("LLM_API_KEY")
            self.base_url = base_url or os.getenv("LLM_BASE_URL")
        else:
            # 根据provider确定API密钥和base_url
            self.api_key, self.base_url = self._resolve_credentials(
                api_key, base_url)

        # 验证必要参数
        if not self.model:
            self.model = self._get_default_model()
        if not all([self.api_key, self.base_url]):
            raise HelloAgentsException("API密钥和服务地址必须被提供或在.env文件中定义。")

        # 创建OpenAI客户端
        self._client = self._create_client()

    def _auto_detect_provider(self, api_key: Optional[str], base_url: Optional[str]) -> str:
        """
        自动检测LLM提供商

        检测逻辑：
        1. 优先检查特定提供商的环境变量
        2. 根据API密钥格式判断
        3. 根据base_url判断
        4. 默认返回通用配置
        """
        # 1. 检查特定提供商的环境变量
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("DEEPSEEK_API_KEY"):
            return "deepseek"
        if os.getenv("DASHSCOPE_API_KEY"):
            return "qwen"
        if os.getenv("MODELSCOPE_API_KEY"):
            return "modelscope"
        if os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY"):
            return "kimi"
        if os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY"):
            return "zhipu"
        if os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_HOST"):
            return "ollama"
        if os.getenv("VLLM_API_KEY") or os.getenv("VLLM_HOST"):
            return "vllm"

        # 2. 根据API密钥格式判断
        actual_api_key = api_key or os.getenv("LLM_API_KEY")
        if actual_api_key:
            actual_key_lower = actual_api_key.lower()
            if actual_api_key.startswith("ms-"):
                return "modelscope"
            elif actual_key_lower == "ollama":
                return "ollama"
            elif actual_key_lower == "vllm":
                return "vllm"
            elif actual_key_lower == "local":
                return "local"
            elif actual_api_key.startswith("sk-") and len(actual_api_key) > 50:
                # 可能是OpenAI、DeepSeek或Kimi，需要进一步判断
                pass
            elif actual_api_key.endswith(".") or "." in actual_api_key[-20:]:
                # 智谱AI的API密钥格式通常包含点号
                return "zhipu"

        # 3. 根据base_url判断
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")
        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            if "api.openai.com" in base_url_lower:
                return "openai"
            elif "api.deepseek.com" in base_url_lower:
                return "deepseek"
            elif "dashscope.aliyuncs.com" in base_url_lower:
                return "qwen"
            elif "api-inference.modelscope.cn" in base_url_lower:
                return "modelscope"
            elif "api.moonshot.cn" in base_url_lower:
                return "kimi"
            elif "open.bigmodel.cn" in base_url_lower:
                return "zhipu"
            elif "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                # 本地部署检测 - 优先检查特定服务
                if ":11434" in base_url_lower or "ollama" in base_url_lower:
                    return "ollama"
                elif ":8000" in base_url_lower and "vllm" in base_url_lower:
                    return "vllm"
                elif ":8080" in base_url_lower or ":7860" in base_url_lower:
                    return "local"
                else:
                    # 根据API密钥进一步判断
                    if actual_api_key and actual_api_key.lower() == "ollama":
                        return "ollama"
                    elif actual_api_key and actual_api_key.lower() == "vllm":
                        return "vllm"
                    else:
                        return "local"
            elif any(port in base_url_lower for port in [":8080", ":7860", ":5000"]):
                # 常见的本地部署端口
                return "local"

        # 4. 默认返回auto，使用通用配置
        return "auto"

    def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple[str, str]:
        """根据provider解析API密钥和base_url"""
        if self.provider == "openai":
            resolved_api_key = api_key or os.getenv(
                "OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv(
                "LLM_BASE_URL") or "https://api.openai.com/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "deepseek":
            resolved_api_key = api_key or os.getenv(
                "DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv(
                "LLM_BASE_URL") or "https://api.deepseek.com"
            return resolved_api_key, resolved_base_url

        elif self.provider == "qwen":
            resolved_api_key = api_key or os.getenv(
                "DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv(
                "LLM_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "modelscope":
            resolved_api_key = api_key or os.getenv(
                "MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv(
                "LLM_BASE_URL") or "https://api-inference.modelscope.cn/v1/"
            return resolved_api_key, resolved_base_url

        elif self.provider == "kimi":
            resolved_api_key = api_key or os.getenv("KIMI_API_KEY") or os.getenv(
                "MOONSHOT_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv(
                "LLM_BASE_URL") or "https://api.moonshot.cn/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "zhipu":
            resolved_api_key = api_key or os.getenv("ZHIPU_API_KEY") or os.getenv(
                "GLM_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv(
                "LLM_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4"
            return resolved_api_key, resolved_base_url

        elif self.provider == "ollama":
            resolved_api_key = api_key or os.getenv(
                "OLLAMA_API_KEY") or os.getenv("LLM_API_KEY") or "ollama"
            resolved_base_url = base_url or os.getenv("OLLAMA_HOST") or os.getenv(
                "LLM_BASE_URL") or "http://localhost:11434/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "vllm":
            resolved_api_key = api_key or os.getenv(
                "VLLM_API_KEY") or os.getenv("LLM_API_KEY") or "vllm"
            resolved_base_url = base_url or os.getenv("VLLM_HOST") or os.getenv(
                "LLM_BASE_URL") or "http://localhost:8000/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "local":
            resolved_api_key = api_key or os.getenv("LLM_API_KEY") or "local"
            resolved_base_url = base_url or os.getenv(
                "LLM_BASE_URL") or "http://localhost:8000/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "custom":
            resolved_api_key = api_key or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL")
            return resolved_api_key, resolved_base_url

        else:
            # auto或其他情况：使用通用配置，支持任何OpenAI兼容的服务
            resolved_api_key = api_key or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL")
            return resolved_api_key, resolved_base_url

    def _create_client(self) -> OpenAI:
        """创建OpenAI客户端"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def _get_default_model(self) -> str:
        """获取默认模型"""
        if self.provider == "openai":
            return "gpt-3.5-turbo"
        elif self.provider == "deepseek":
            return "deepseek-chat"
        elif self.provider == "qwen":
            return "qwen-plus"
        elif self.provider == "modelscope":
            return "Qwen/Qwen2.5-72B-Instruct"
        elif self.provider == "kimi":
            return "moonshot-v1-8k"
        elif self.provider == "zhipu":
            return "glm-4"
        elif self.provider == "ollama":
            return "llama3.2"  # Ollama常用模型
        elif self.provider == "vllm":
            return "meta-llama/Llama-2-7b-chat-hf"  # vLLM常用模型
        elif self.provider == "local":
            return "local-model"  # 本地模型占位符
        elif self.provider == "custom":
            return self.model or "gpt-3.5-turbo"
        else:
            # auto或其他情况：根据base_url智能推断默认模型
            base_url = os.getenv("LLM_BASE_URL", "")
            base_url_lower = base_url.lower()
            if "modelscope" in base_url_lower:
                return "Qwen/Qwen2.5-72B-Instruct"
            elif "deepseek" in base_url_lower:
                return "deepseek-chat"
            elif "dashscope" in base_url_lower:
                return "qwen-plus"
            elif "moonshot" in base_url_lower:
                return "moonshot-v1-8k"
            elif "bigmodel" in base_url_lower:
                return "glm-4"
            elif "ollama" in base_url_lower or ":11434" in base_url_lower:
                return "llama3.2"
            elif ":8000" in base_url_lower or "vllm" in base_url_lower:
                return "meta-llama/Llama-2-7b-chat-hf"
            elif "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                return "local-model"
            else:
                return "gpt-3.5-turbo"

    def think(self, messages: list[dict[str, str]], temperature: Optional[float] = None) -> Iterator[str]:
        """
        调用大语言模型进行思考，并返回流式响应。
        这是主要的调用方法，默认使用流式响应以获得更好的用户体验。

        Args:
            messages: 消息列表
            temperature: 温度参数，如果未提供则使用初始化时的值

        Yields:
            str: 流式响应的文本片段
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
                    yield content
            print()  # 在流式输出结束后换行

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            raise HelloAgentsException(f"LLM调用失败: {str(e)}")

    def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        非流式调用LLM，返回完整响应。
        适用于不需要流式输出的场景。
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
            )
            return response.choices[0].message.content
        except Exception as e:
            raise HelloAgentsException(f"LLM调用失败: {str(e)}")

    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        """
        流式调用LLM的别名方法，与think方法功能相同。
        保持向后兼容性。
        """
        temperature = kwargs.get('temperature')
        yield from self.think(messages, temperature)
