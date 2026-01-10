import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from hello_agents.core.tool.base_tool import BaseTool
from hello_agents.core.tool.tool_schema import McpToolInfo
from hello_agents.core.tool.mcp_tool import McpTool
if TYPE_CHECKING:
    from hello_agents.core.agent.agent_schema import AgentContext


class ToolCollection:
    """
    工具集合类 - 管理可用的工具
    严格对齐 Java: ToolCollection
    """

    def __init__(self):
        self.tool_map: Dict[str, BaseTool] = {}
        self.mcp_tool_map: Dict[str, McpToolInfo] = {}
        self.agent_context: Optional["AgentContext"] = None

        # ===== 数字员工相关 =====
        # task 未并发的情况下：
        # 每一个 task 执行时，数字员工列表会更新
        # TODO: 并发情况下需要处理
        self.current_task: Optional[str] = None
        self.digital_employees: Optional[Dict[str, Any]] = None

    # =============================
    # 添加工具
    # =============================
    def add_tool(self, tool: BaseTool) -> None:
        self.tool_map[tool.name] = tool

    # =============================
    # 获取工具
    # =============================
    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self.tool_map.get(name)

    # =============================
    # 添加 MCP 工具
    # =============================
    def add_mcp_tool(
        self,
        name: str,
        desc: str,
        parameters: str,
        mcp_server_url: str,
    ) -> None:
        self.mcp_tool_map[name] = McpToolInfo(
            name=name,
            desc=desc,
            parameters=parameters,
            mcp_server_url=mcp_server_url,
        )

    # =============================
    # 获取 MCP 工具
    # =============================
    def get_mcp_tool(self, name: str) -> Optional[McpToolInfo]:
        return self.mcp_tool_map.get(name)

    # =============================
    # 执行工具
    # =============================
    def execute(self, name: str, tool_input: Any) -> Any:
        if name in self.tool_map:
            tool = self.get_tool(name)
            return tool.execute(tool_input)

        elif name in self.mcp_tool_map:
            tool_info = self.mcp_tool_map.get(name)

            mcp_tool = McpTool()
            mcp_tool.agent_context = self.agent_context

            return mcp_tool.call_tool(
                tool_info.mcp_server_url,
                name,
                tool_input,
            )

        else:
            self.agent_context.printer.send("Error: Unknown tool %s", name)

        return None

    # =============================
    # 设置数字员工
    # =============================
    def update_digital_employee(self, digital_employee: Optional[Dict[str, Any]]) -> None:
        if digital_employee is None:
            self.agent_context.printer.send(
                "requestId:%s setDigitalEmployee: %s",
                self.agent_context.request_id if self.agent_context else None,
                digital_employee,
            )
        self.digital_employees = digital_employee

    # =============================
    # 获取数字员工名称
    # =============================
    def get_digital_employee(self, tool_name: Optional[str]) -> Optional[str]:
        if not tool_name:
            return None

        if self.digital_employees is None:
            return None

        return self.digital_employees.get(tool_name)

    def set_current_task(self, task: str):
        self.current_task = task
