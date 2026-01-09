from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any

from hello_agents.core.printer.printer import Printer
from hello_agents.core.tool.tool_collection import ToolCollection


@dataclass
class AgentContext:
    # ========= 基础追踪信息 =========
    request_id: str
    session_id: Optional[str] = None

    # ========= 用户 & 任务语义 =========
    query: Optional[str] = None           # 用户原始问题
    task: Optional[str] = None            # 当前 Agent 子任务
    agent_type: Optional[int] = None      # AgentType 枚举值

    # ========= 流式与输出控制 =========
    printer: Optional[Printer] = None         # Printer 实例（SSE / WS / Console）
    is_stream: bool = False
    stream_message_type: str = "llm"      # 与 Java 对齐：context.getStreamMessageType()

    # ========= 工具与能力 =========
    tool_collection: Optional[ToolCollection] = None  # ToolCollection

    # ========= Prompt & 模板 =========
    sop_prompt: Optional[str] = None
    base_prompt: Optional[str] = None
    template_type: Optional[str] = None   # markdown / text / card

    # ========= 文件 & RAG =========
    product_files: List[Any] = field(default_factory=list)
    task_product_files: List[Any] = field(default_factory=list)

    # ========= 环境信息 =========
    date_info: Optional[str] = None


class AgentType(Enum):
    """agent类型"""
    COMPREHENSIVE = 1
    WORKFLOW = 2
    PLAN_SOLVE = 3
    ROUTER = 4
    REACT = 5

    @staticmethod
    def from_code(value: int) -> "AgentType":
        """等价于 Java 的 fromCode(int value)"""
        for agent_type in AgentType:
            if agent_type.value == value:
                return agent_type
        raise ValueError(f"Invalid AgentType code: {value}")
