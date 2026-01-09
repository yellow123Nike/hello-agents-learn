from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RoleType(Enum):
    """消息角色类型枚举"""
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"

    @staticmethod
    def is_valid(role: str) -> bool:
        return role in {r.value for r in RoleType}

    @staticmethod
    def from_string(role: str) -> "RoleType":
        for r in RoleType:
            if r.value == role:
                return r
        raise ValueError(f"Invalid role: {role}")


class LLMParams(BaseModel):
    """LLM的可配置参数"""
    # 模型基础配置
    model_name: str = Field(description="模型名称")
    api_key: str = Field(description="APIkey")
    base_url: str = Field(description="API URL")
    # 模型生成行为配置
    temperature: Optional[float] = Field(
        default=0.7, description="生成随机性:(0:确定性输出,0.7:平衡随机性,2:高度随机)")
    max_tokens: Optional[int] = Field(
        default=8096, description="限制生成的最大 token数")
    # 输出配置
    n: Optional[int] = Field(
        default=1, description="生成多个候选回复(默认 1,增加会提高成本)")
    stream: Optional[bool] = Field(
        default=False, description="是否流式传输结果（实时逐字返回，适用于聊天界面)")
    is_claude: Optional[bool] = Field(
        default=False, description="claude和openai在tool_call方面存在一定差异")
