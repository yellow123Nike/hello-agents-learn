
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from hello_agents.core.tool.tool_schema import ToolCall


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


@dataclass
class ProviderSpec:
    name: str
    max_tokens_field: str
    supports_tool_calls: bool = True
    tool_call_style: Literal["openai", "claude"] = "openai"


PROVIDERS = {
    "openai": ProviderSpec(
        name="openai",
        max_tokens_field="max_tokens",
    ),
    "gpt5": ProviderSpec(
        name="gpt5",
        max_tokens_field="max_com_tokens",
    ),
    "claude": ProviderSpec(
        name="claude",
        max_tokens_field="max_tokens",
        tool_call_style="claude",
    ),
}


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


class ToolChoice(Enum):
    """
    none:明确禁止模型调用任何工具
    auto:由模型自行判断是否需要调用工具  vLLM 默认不支持 auto，除非显式开启 --enable-auto-tool-choice --tool-call-parser
    required:强制模型必须调用工具（至少一次）
    """

    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"

    @classmethod
    def is_valid(cls, value: "ToolChoice") -> bool:
        return value is not None and value.value in {
            "none", "auto", "required"
        }

    @classmethod
    def from_string(cls, value: str) -> "ToolChoice":
        try:
            return ToolChoice(value)
        except ValueError:
            raise ValueError(f"Invalid tool choice: {value}")


class FunctionCallType(Enum):
    STRUCT_PARSE = "struct_parse"  # 把工具调用当成文本结构解析问题
    FUNCTION_CALL = "function_call"  # 让模型走 OpenAI 原生的工具调用协议

    @classmethod
    def is_valid(cls, value: "FunctionCallType") -> bool:
        return value in cls


@dataclass
class ToolCallResponse:
    content: Optional[str]
    tool_calls: List[ToolCall]
    finish_reason: Optional[str] = None
    total_tokens: Optional[int] = None
    duration: Optional[float] = None
