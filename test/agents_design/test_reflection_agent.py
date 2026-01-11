import ast
import asyncio
import math
import os

import pytest
from hello_agents.agents_design.reflection_agent import ReflectionAgent
from hello_agents.core.agent.agent_schema import AgentContext, AgentRequest
from hello_agents.core.llm.llm import LLMClient
from hello_agents.core.llm.llm_schema import LLMParams
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


@pytest.mark.asyncio
async def test_react_impl_agent():
    tools = ToolCollection()
    io_printer = StdoutPrinter(request=AgentRequest(request_id="test-fc"))
    context = AgentContext(
        request_id="test-fc",
        base_prompt=(
            "我的反思助手"
        ),
        date_info="2025-01-01",
        printer=io_printer,
    )
    context.tool_collection = tools
    agent = ReflectionAgent(context, LLMClient(params, provider_name))
    query = (
        "写一篇关于人工智能发展历程的简短文章"
    )
    final_answer = await agent.run(query)
    print("\n===== FINAL ANSWER =====")
    print(final_answer)
    print("\n===== Message =====")
    print(agent.memory.messages)

asyncio.run(test_react_impl_agent())
