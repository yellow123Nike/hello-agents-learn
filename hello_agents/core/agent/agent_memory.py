from typing import Iterable, List, Optional
from enum import Enum

from hello_agents.core.llm.llm_schema import RoleType
from hello_agents.core.llm.message import Message


class Memory:
    """
    记忆类 - 管理代理的消息历史
    """

    def __init__(self):
        self.messages: List[Message] = []

    # ----------------------------
    # 添加消息
    # ----------------------------
    def add_message(self, message:  Message | Iterable[Message]):
        if isinstance(message, Message):
            self.messages.append(message)
        else:
            self.messages.extend(message)

    # ----------------------------
    # 读取消息
    # ----------------------------

    def get_last_message(self):
        return self.messages[-1] if self.messages else None

    def get(self, index: int):
        return self.messages[index]

    def size(self):
        return len(self.messages)

    def is_empty(self):
        return not self.messages

    # ----------------------------
    # 清理逻辑
    # ----------------------------
    def clear(self) -> None:
        self.messages.clear()

    def clear_tool_context(self):
        """
        清空工具执行历史，包括：
        1. role == TOOL 的消息
        2. ASSISTANT 且包含 tool_calls 的消息
        3. 特定前缀的 planning / reflection 消息
        """
        filtered_messages = []
        for message in self.messages:
            # 1. 移除 TOOL 消息
            if message.role == RoleType.TOOL:
                continue
            # 2. 移除带 tool_calls 的 ASSISTANT 消息
            if (
                message.role == RoleType.ASSISTANT
                and message.tool_calls
            ):
                continue
            # 3. 移除特定 planning 文本
            if (
                message.content is not None
                and message.content.startswith("根据当前状态和可用工具，确定下一步行动")
            ):
                continue

            filtered_messages.append(message)

        self.messages = filtered_messages

    # ----------------------------
    # 格式化输出
    # ----------------------------
    def get_format_message(self) -> str:
        """
        返回格式化后的 message 字符串
        """
        lines = []
        for message in self.messages:
            lines.append(
                f"role:{message.role} content:{message.content}"
            )
        return "\n".join(lines)
