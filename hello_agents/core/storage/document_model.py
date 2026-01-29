"""存储文档形态：抽象为 Pydantic BaseModel，与 Message 互转。"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from hello_agents.core.llm.llm_schema import RoleType
from hello_agents.core.llm.message import Message
from hello_agents.core.tool.tool_schema import ToolCall


class DocumentModel(BaseModel):
    """
    存储文档形态的抽象：id、content、metadata（子类可扩展如 embedding、score）。
    与 Message 互转，供 BaseStore 及子类统一使用。
    """

    id: str = Field(description="文档唯一标识")
    content: str = Field(default="", description="文档正文")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据（如 role、base64_image、tool_calls 等）")
    score: Optional[float] = Field(default=None, description="检索得分（仅检索结果时有值）")
    model_config = {"extra": "allow"}

    # -------------------------------------------------------------------------
    # 与 Message 互转
    # -------------------------------------------------------------------------

    @classmethod
    def from_message(cls, msg: Message, doc_id: Optional[str] = None) -> "DocumentModel":
        """从 Message 构造文档。"""
        id_ = doc_id or str(uuid.uuid4())
        content = msg.content or ""
        meta: Dict[str, Any] = {}
        if msg.role is not None:
            meta["role"] = msg.role.value
        if msg.base64_image is not None:
            meta["base64_image"] = msg.base64_image
        if msg.tool_call_id is not None:
            meta["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            meta["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": (
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        if tc.function else None
                    ),
                }
                for tc in msg.tool_calls
            ]
        return cls(id=id_, content=content, metadata=meta)

    def to_message(self) -> Message:
        """转为 Message。"""
        meta = self.metadata or {}
        role = meta.get("role")
        if role is not None and isinstance(role, str):
            try:
                role = RoleType(role)
            except ValueError:
                role = None
        base64_image = meta.get("base64_image")
        tool_call_id = meta.get("tool_call_id")
        tool_calls_raw = meta.get("tool_calls") or []
        tool_calls: List[ToolCall] = []
        for t in tool_calls_raw:
            if isinstance(t, dict):
                fn = t.get("function")
                func = (
                    ToolCall.Function(
                        name=fn.get("name") if isinstance(fn, dict) else None,
                        arguments=fn.get("arguments") if isinstance(fn, dict) else None,
                    )
                    if fn else None
                )
                tool_calls.append(
                    ToolCall(id=t.get("id"), type=t.get("type"), function=func)
                )
        return Message(
            role=role,
            content=self.content,
            base64_image=base64_image,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls if tool_calls else None,
        )

    # -------------------------------------------------------------------------
    # 与 dict 互转（兼容子类存储层）
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """转为存储用 dict（id、content、metadata；不含 score）。"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DocumentModel":
        """从 dict 构造（如 get/search 返回的原始结构）。"""
        return cls(
            id=d.get("id", ""),
            content=d.get("content", ""),
            metadata=d.get("metadata") or {},
            score=d.get("score"),
        )
