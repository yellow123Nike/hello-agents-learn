import ast
import asyncio
import math
import os

import pytest
from hello_agents.agents_design.plan_solve import PlanSolveAgent
from hello_agents.agents_design.react_agent import ReactAgent
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
            """你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
                    你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
                    请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。
                    无工具时，结合自身能力回答问题
                    """

        ),
        sop_prompt=(
            "规划执行助手。\n"
        ),
        date_info="2025-01-01",
        printer=io_printer,
    )
    context.tool_collection = tools
    agent = PlanSolveAgent(context,  LLMClient(
        params, provider_name), is_close_update=False)
    query = (
        "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    )
    final_answer = await agent.run(query)
    print("\n===== FINAL ANSWER =====")
    print(final_answer)
    print("\n===== Message =====")
    print(agent.memory.messages)

asyncio.run(test_react_impl_agent())
