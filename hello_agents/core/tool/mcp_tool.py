import json
import logging
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Any, Dict, Optional
from hello_agents.core.tool.base_tool import BaseTool
from hello_agents.core.tool.http_util import OkHttpUtil
if TYPE_CHECKING:
    from hello_agents.core.agent.agent_schema import AgentContext


class McpTool(BaseTool):
    """
    MCP 工具实现类
    """

    def __init__(self, agent_context: "AgentContext"):
        self.agent_context = agent_context

    # =============================
    # 内部 DTO：McpToolRequest
    # =============================
    @dataclass
    class McpToolRequest:
        server_url: Optional[str] = None
        name: Optional[str] = None
        arguments: Optional[Dict[str, Any]] = None

        def to_dict(self) -> Dict[str, Any]:
            return {
                k: v for k, v in {
                    "server_url": self.server_url,
                    "name": self.name,
                    "arguments": self.arguments,
                }.items() if v is not None
            }

    # =============================
    # 内部 DTO：McpToolResponse
    # =============================
    @dataclass
    class McpToolResponse:
        code: Optional[str] = None
        message: Optional[str] = None
        data: Optional[str] = None

    # =============================
    # BaseTool 接口实现
    # =============================
    def get_name(self) -> str:
        return "mcp_tool"

    def get_description(self) -> str:
        return ""

    def to_params(self) -> Dict[str, Any]:
        return {}

    def execute(self, input: Any) -> Any:
        """
        约定：
        input = {
            "server_url": "...",
            "tool_name": "...",
            "arguments": {...}
        }
        """
        if not isinstance(input, dict):
            raise ValueError("McpTool.execute input must be a dict")

        server_url = input.get("server_url")
        tool_name = input.get("tool_name")
        arguments = input.get("arguments", {})

        if not server_url or not tool_name:
            raise ValueError("server_url and tool_name are required")

        return self.call_tool(server_url, tool_name, arguments)

    # =============================
    # 业务方法：listTool
    # =============================
    def list_tool(self, mcp_server_url: str) -> str:
        try:
            MCP_CLIENT_URL = os.getenv("MCP_CLIENT_URL")
            mcp_client_url = f"{MCP_CLIENT_URL}/v1/tool/list"

            request = McpTool.McpToolRequest(
                server_url=mcp_server_url
            )

            payload = json.dumps(request.to_dict(), ensure_ascii=False)

            response = OkHttpUtil.post_json(
                url=mcp_client_url,
                body=payload,
                headers=None,
                timeout=30,
            )

            self.agent_context.printer.send(
                "list tool request: %s response: %s",
                payload,
                response,
            )
            return response

        except Exception as e:
            self.agent_context.printer.send(
                "%s list tool error",
                self.agent_context.request_id,
                exc_info=e,
            )
            return ""

    # =============================
    # 业务方法：callTool
    # =============================
    def call_tool(
        self,
        mcp_server_url: str,
        tool_name: str,
        input: Dict[str, Any],
    ) -> str:
        try:
            MCP_CLIENT_URL = os.getenv("MCP_CLIENT_URL")
            mcp_client_url = f"{MCP_CLIENT_URL}/v1/tool/call"

            request = McpTool.McpToolRequest(
                name=tool_name,
                server_url=mcp_server_url,
                arguments=input,
            )

            payload = json.dumps(request.to_dict(), ensure_ascii=False)

            response = OkHttpUtil.post_json(
                url=mcp_client_url,
                body=payload,
                headers=None,
                timeout=30,
            )

            self.agent_context.printer.send(
                "call tool request: %s response: %s",
                payload,
                response,
            )
            return response

        except Exception as e:
            self.agent_context.printer.send(
                "%s call tool error",
                self.agent_context.request_id,
                exc_info=e,
            )
            return ""
