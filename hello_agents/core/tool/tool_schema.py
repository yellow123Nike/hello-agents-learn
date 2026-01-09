
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolCall:
    """
    工具调用出参
    """
    id: Optional[str] = None
    type: Optional[str] = None
    function: Optional["ToolCall.Function"] = None

    @dataclass
    class Function:
        """
        函数信息类
        """
        name: Optional[str] = None
        arguments: Optional[str] = None


@dataclass
class McpToolInfo:
    mcp_server_url: Optional[str] = None   # MCP Server 地址
    name: Optional[str] = None             # 工具名称
    desc: Optional[str] = None             # 工具描述
    parameters: Optional[str] = None       # 参数定义（通常是 JSON Schema / 描述串）
