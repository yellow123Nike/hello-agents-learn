import asyncio
import json
from typing import List, Optional
from pydantic import Field
from hello_agents.core.agent.agent_memory import Memory
from hello_agents.core.agent.agent_schema import AgentContext, AgentState
from hello_agents.core.llm.llm import LLMClient
from hello_agents.core.llm.llm_schema import RoleType, ToolCall
from hello_agents.core.llm.message import Message
from hello_agents.core.tool.tool_collection import ToolCollection

# ===== BaseAgent =====


class BaseAgent:
    """Agent 基类"""

    def __init__(
        self,
        name: str = Field(description="Agent 的唯一名称，用于标识、日志记录与调度"),
        description: str = Field(description="Agent 的职责说明（人类可读，不直接参与推理）"),
        system_prompt: str = Field(
            description="系统级 Prompt,用于定义 Agent 的角色、能力边界与全局行为约束"),
        next_step_prompt: str = Field(
            description="单步推理引导 Prompt，用于驱动 Agent 决定“下一步做什么”"),
        llm: Optional[LLMClient] = Field(
            description="Agent 绑定的 LLM 实例，负责实际推理与文本生成"),
        context: Optional[AgentContext] = Field(
            description="Agent 运行上下文，用于保存状态、历史对话及中间结果"),
        max_steps: int = Field(
            default=10, description="Agent 允许执行的最大推理步数，用于防止无限循环"),
        duplicate_threshold: int = Field(
            default=2, description="连续重复输出或决策的阈值，用于检测 Agent 是否陷入死循环"),
    ):
        # core
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.next_step_prompt = next_step_prompt
        # 角色级 / 组织级配置  --应该从配置中来
        self.digital_employee_prompt: Optional[str] = None
        self.available_tools = ToolCollection()
        self.memory = Memory()
        self.llm = llm
        self.context = context

        # 执行控制
        self.state = AgentState.IDLE
        self.max_steps = max_steps
        self.current_step = 0
        self.duplicate_threshold = duplicate_threshold

    # ===== abstract step =====
    async def step(self):
        """
        执行单个 Agent 推理步骤
        """
        raise NotImplementedError

    # ===== main loop =====
    async def run(self, query: str):
        self.state = AgentState.IDLE
        self.current_step = 0

        if query:
            self.update_memory(RoleType.USER, query)

        results: List[str] = []

        try:
            while self.current_step < self.max_steps and self.state != AgentState.FINISHED:
                self.current_step += 1
                req_id = self.context.request_id if self.context else "-"
                self.context.printer.send(
                    f"{req_id} {self.name} Executing step {self.current_step}/{self.max_steps}")

                step_result = await self.step()
                results.append(step_result)

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(
                    f"Terminated: Reached max steps ({self.max_steps})")

        except Exception:
            self.state = AgentState.ERROR
            raise

        return results[-1] if results else "No steps executed"

    # ===== memory =====
    def update_memory(
        self,
        role: RoleType,
        content: str,
        base64_image: Optional[str] = None,
        *args,
    ):
        if role == RoleType.USER:
            msg = Message.user_message(content, base64_image)
        elif role == RoleType.SYSTEM:
            msg = Message.system_message(content, base64_image)
        elif role == RoleType.ASSISTANT:
            msg = Message.assistant_message(content, base64_image)
        elif role == RoleType.TOOL:
            msg = Message.tool_message(content, args[0], base64_image)
        else:
            raise ValueError(f"Unsupported role type: {role}")

        self.memory.add_message(msg)

    # ===== tool execution =====
    async def execute_tool(self, command: ToolCall):
        try:
            func = command.function
            if not func or not func.name:
                return "Error: Invalid function call format"

            name = func.name
            args = json.loads(func.arguments or "{}")

            result = await self.available_tools.execute(name, args)
            req_id = self.context.request_id if self.context else "-"
            self.context.printer.send(
                f"{req_id} execute tool: {name} {args} result {result}")

            return str(result) if result is not None else ""

        except Exception as e:
            req_id = self.context.request_id if self.context else "-"
            self.context.printer.send(
                f"{req_id} execute tool {name if 'name' in locals() else '-'} failed: {e}")
            return f"Tool {name if 'name' in locals() else ''} Error."

    async def execute_tools(self, commands: List[ToolCall]):
        """
        并发执行多个工具调用命令并返回执行结果
        :param commands: 工具调用命令列表
        :return: key 为 tool_call.id，value 为执行结果
        """

        async def _run_tool(tool_call: ToolCall):
            try:
                result = await self.execute_tool(tool_call)
                return tool_call.id, result
            except Exception as e:
                return tool_call.id, f"Tool Error: {e}"

        tasks = [asyncio.create_task(_run_tool(cmd)) for cmd in commands]

        results = await asyncio.gather(*tasks)

        return {tool_id: result for tool_id, result in results}
