import json
from typing import Iterable, List, Optional, Dict, Any
from enum import Enum

from hello_agents.core.llm.llm_schema import RoleType
from hello_agents.core.llm.message import Message
from hello_agents.core.llm.string_util import text_desensitization
from hello_agents.core.llm.llm_prompt import SENSITIVE_PATTERNS


class Memory:
    """
    记忆类 - 管理单次 Agent 会话的对话历史记录
    
    职责：
    - 存储当前会话的 Message 列表（USER、ASSISTANT、SYSTEM、TOOL）
    - 用于构建 LLM 的输入上下文
    - 临时性存储，会话结束后清空
    - 不持久化，不跨会话
    - 与单次 agent_id/request_id 相关，每个 Agent 实例独立
    
    注意：与 MemoryTool 的区别
    - Memory: 单次 agent_id 的对话历史（临时、内存中、不持久化）
    - MemoryTool: 与 user_id 相关的持久化记忆（可索引、可衰减、可持久化、跨会话）
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

    # ----------------------------
    # 获取系统指令
    # ----------------------------
    def get_system_prompt(self) -> str:
        for message in self.messages:
            if message.role == RoleType.SYSTEM:
                return message.content
        return ""
    
    #----------------------------
    # 获取非系统指令的消息的最近n条
    # ----------------------------
    def get_non_system_prompt(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        获取非系统指令的消息的最近n条，返回格式化的字典列表
        
        注意：此方法用于上下文构建，返回通用格式（GPT风格），不依赖特定的provider
        
        Args:
            top_n: 获取最近n条消息（不包括系统消息）
            
        Returns:
            格式化的消息字典列表，每个字典包含 role 和 content 等字段
        """
        # 过滤掉系统消息，获取最近top_n条
        non_system_messages = [msg for msg in self.messages if msg.role != RoleType.SYSTEM]
        messages = non_system_messages[-top_n:] if len(non_system_messages) > top_n else non_system_messages
        
        formatted = []
        for msg in messages:
            message_map: Dict[str, Any] = {}
            
            # ===== multimodal =====
            # 处理 base64 图像
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
                    "text": msg.content or ""
                })
                message_map["role"] = msg.role.value
                message_map["content"] = multimodal

            # ===== tool calls =====
            # 使用通用格式（GPT风格），不依赖特定provider
            elif msg.tool_calls:
                message_map["role"] = msg.role.value
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
                # 如果有content，也包含进去
                if msg.content:
                    message_map["content"] = msg.content

            # ===== tool result =====
            # 工具执行结果
            elif msg.tool_call_id:
                # 执行结果脱敏
                content = text_desensitization(
                    msg.content or "",
                    SENSITIVE_PATTERNS, 
                )
                message_map["role"] = msg.role.value
                message_map["content"] = content
                message_map["tool_call_id"] = msg.tool_call_id

            # ===== normal text =====
            else:
                message_map["role"] = msg.role.value
                message_map["content"] = msg.content or ""

            formatted.append(message_map)
        return formatted