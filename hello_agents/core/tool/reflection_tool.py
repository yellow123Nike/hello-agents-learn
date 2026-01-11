from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from hello_agents.agents_design.com.planner import Planner
from hello_agents.core.agent.agent_schema import AgentContext
from hello_agents.core.tool.base_tool import BaseTool


@dataclass
class ReflectionTool(BaseTool):
    """
    反思工具类
    """
    agent_context: Optional[AgentContext] = None
    name = "reflection"
    description = """
                这是一个反馈收集器，请将你反馈请求填写到这里
            """

    # ---------- 参数 Schema ----------
    def to_params(self):
        out = {
            "type": "object",
            "properties": {
                "reflection_status": {
                    "description": "是否有反馈意见",
                    "type": "string",
                    "enum": ["yes", "no"]
                },
                "reflection_feedback": {
                    "description": "反馈意见,在reflection_status为 yes时必填",
                    "type": "string",
                },
            },
            "required": ["command"]
        }
        return out

    # ---------- 执行入口 ----------

    async def execute(self, input: Any):
        if not isinstance(input, dict):
            raise ValueError("Input must be a Map")

        return input
