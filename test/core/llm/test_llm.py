
import os
import pytest
from hello_agents.core.agent.agent_schema import AgentContext, AgentRequest
from hello_agents.core.llm.llm import LLMClient
from hello_agents.core.llm.llm_schema import FunctionCallType, LLMParams, RoleType, ToolChoice
from hello_agents.core.llm.message import Message
from hello_agents.core.printer.stdout_printer import StdoutPrinter
from hello_agents.core.tool.base_tool import BaseTool
from hello_agents.core.tool.tool_collection import ToolCollection
from dotenv import load_dotenv
load_dotenv()
provider_name = "openai"
params = LLMParams(
    model_name=os.getenv("LLM_MODEL_ID", "gpt-4o"),
    api_key=os.getenv("LLM_API_KEY", "sk-"),
    base_url=os.getenv("LLM_BASE_URL", "http://0.0.0.0:18006/v1/"),
    temperature=0.7,
    max_tokens=8024,
    is_claude=False,
)
messages = [
    Message(
        role=RoleType.USER,
        content="Hello",
    )
]
context = AgentContext(request_id="test-llm-id")
io_printer = StdoutPrinter(request=AgentRequest(request_id="test-fc"))


@pytest.mark.asyncio
async def test_ask_llm_non_stream():
    llm = LLMClient(params, provider_name)
    result = await llm.ask_llm_once(
        context=context,
        messages=messages,
        system_msgs=None,
    )
    print(result)


@pytest.mark.asyncio
async def test_ask_llm_stream():
    llm = LLMClient(params, provider_name)
    async for chunk in llm.ask_llm_stream(
        context=context,
        messages=messages,
        system_msgs=None,
    ):
        print(chunk, end="", flush=True)


class DummyAddTool(BaseTool):
    name = "add"
    description = "Add two numbers"

    def to_params(self):
        return {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        }

    def execute(self, tool_input):
        return tool_input["a"] + tool_input["b"]


@pytest.mark.asyncio
async def test_ask_tool_function_call():
    tools = ToolCollection()
    tools.add_tool(DummyAddTool())
    context = AgentContext(request_id="test-fc", printer=io_printer)
    messages = [Message(role=RoleType.USER, content="请使用工具，Add 1 and 2")]
    system_msgs = Message(role=RoleType.SYSTEM, content="AI助手")
    llm = LLMClient(params, provider_name)
    # ===== Act =====
    result = await llm.ask_tool(
        context=context,
        messages=messages,
        tools=tools,
        tool_choice=ToolChoice.AUTO,
        system_msgs=system_msgs,
        function_call_type=FunctionCallType.STRUCT_PARSE,
    )
    print(result)
