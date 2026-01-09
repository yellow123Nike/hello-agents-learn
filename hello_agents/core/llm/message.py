from dataclasses import dataclass
from typing import List, Optional

from hello_agents.core.llm.llm_schema import RoleType
from hello_agents.core.tool.tool_schema import ToolCall



@dataclass
class Message:
    """
    消息类 - 表示代理系统中的各种消息
    """
    role: Optional[RoleType] = None          # 消息角色
    content: Optional[str] = None            # 消息内容
    base64_image: Optional[str] = None       # 图片数据（base64 编码）
    tool_call_id: Optional[str] = None       # 工具调用 ID
    tool_calls: Optional[List[ToolCall]] = None  # 工具调用列表

    @staticmethod
    def user_message(content: str, base64_image: Optional[str] = None):
        return Message(
            role=RoleType.USER,
            content=content,
            base64_image=base64_image,
        )

    @staticmethod
    def system_message(content: str, base64_image: Optional[str] = None):
        return Message(
            role=RoleType.SYSTEM,
            content=content,
            base64_image=base64_image,
        )

    @staticmethod
    def assistant_message(content: str, base64_image: Optional[str] = None):
        return Message(
            role=RoleType.ASSISTANT,
            content=content,
            base64_image=base64_image,
        )

    @staticmethod
    def tool_message(
        content: str,
        tool_call_id: str,
        base64_image: Optional[str] = None,
    ):
        return Message(
            role=RoleType.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            base64_image=base64_image,
        )

    @staticmethod
    def from_tool_calls(
        content: str,
        tool_calls: List[ToolCall],
    ):
        return Message(
            role=RoleType.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        )
