from datetime import datetime
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    """记忆项数据结构 - 用于 MemoryTool 的持久化记忆存储,用于数据库的存储结构"""
    
    id: str = Field(description="记忆项的唯一标识符，用于索引和检索")
    content: str = Field(description="记忆内容，存储实际的交互信息、知识或经验")
    memory_type: Literal["working", "episodic", "semantic", "perceptual"] = Field(
        description="记忆类型：working(工作记忆-短期)、episodic(情景记忆-事件)、semantic(语义记忆-知识)、perceptual(感知记忆-多模态)"
    )
    user_id: str = Field(description="用户ID，记忆与用户绑定，支持跨会话持久化")
    timestamp: datetime = Field(description="记忆创建或更新时间戳，用于时间衰减和排序")
    importance: float = Field(
        default=0.5,
        description="重要性分数，范围0.0-1.0，用于记忆衰减和优先级排序（0.0=不重要，1.0=非常重要）"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="扩展元数据字典，可存储session_id、source、tags等额外信息"
    )

    class Config:
        arbitrary_types_allowed = True