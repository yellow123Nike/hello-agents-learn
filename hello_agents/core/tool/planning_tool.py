from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from hello_agents.agents_design.com.planner import Planner
from hello_agents.core.agent.agent_schema import AgentContext
from hello_agents.core.tool.base_tool import BaseTool


@dataclass
class PlanningTool(BaseTool):
    """
    计划工具类
    """
    agent_context: Optional[AgentContext] = None
    plan: Optional[Planner] = None
    command_handlers: Dict[str, Callable[[Dict[str, Any]], str]] = field(
        default_factory=dict
    )

    def __post_init__(self):
        self.description = """
                这是一个计划工具，可让代理创建和管理用于解决复杂任务的计划。
                该工具提供创建计划、更新计划步骤和跟踪进度的功能。
                创建计划时，需要创建出有依赖关系的计划，
                计划列表格式如下：
                [执行顺序+编号、任务短标题：任务的细节描述]，
                样式示例如下：[
                    执行顺序1. 任务短标题: 任务描述xxx ..., 
                    执行顺序2. 任务短标题：任务描述xxx ..., 
                    执行顺序3. 任务短标题：任务描述xxx ... ]
            """
        self.name = "planning"
        self.command_handlers = {
            "create": self.create_plan,
            "update": self.update_plan,
            "mark_step": self.mark_step,
            "finish": self.finish_plan,
        }

    # ---------- 参数 Schema ----------
    def to_params(self):
        out = {
            "type": "object",
            "properties": {
                "step_status": {
                    "description": "步骤的执行状态，当command是 mark_step 时使用.- not_started：未开始，- in_progress：进行中，- completed：已完成，- blocked：被阻塞",
                    "type": "string",
                    "enum": ["not_started", "in_progress", "completed", "blocked"]
                },
                "step_notes": {
                    "description": "步骤执行的结果，当command 是 mark_step 时，是备选参数。",
                    "type": "string"},
                "step_index": {
                    "description": "需要更新的步骤索引（从 0 开始计数）。在 command 为 mark_step 时必填",
                    "type": "integer"},
                "title": {
                    "description": "计划的标题。在 command 为 create 时必填，在 update 时为可选参数",
                    "type": "string"},
                "steps": {
                    "description": "计划的任务步骤列表。在 command 为 create 时必填。任务列表的的格式如下：[\"执行顺序 + 编号、执行任务简称：执行任务的细节描述\"]。不同的子任务之间不能重复、也不能交叠，可以收集多个方面的信息，收集信息、查询数据等此类多次工具调用，是可以并行的任务。具体的格式示例如下：- 任务列表示例1: [\"执行顺序1. 执行任务简称（不超过6个字）：执行任务的细节描述（不超过50个字）\", \"执行顺序2. xxx（不超过6个字）：xxx（不超过50个字）, ...\"]；",
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "command": {
                    "description": "需要执行的计划操作指令，可选值：create（创建计划）、update（更新计划）、mark_step（标记步骤状态）、finish（完成计划）",
                    "type": "string",
                    "enum": ["create", "update", "mark_step", "finish"]}
            },
            "required": ["command"]
        }
        return out

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": self._get_properties(),
            "required": ["command"],
        }

    def _get_properties(self) -> Dict[str, Any]:
        return {
            "command": self._get_command_property(),
            "title": self._get_title_property(),
            "steps": self._get_steps_property(),
            "step_index": self._get_step_index_property(),
            "step_status": self._get_step_status_property(),
            "step_notes": self._get_step_notes_property(),
        }

    def _get_command_property(self) -> Dict[str, Any]:
        return {
            "type": "string",
            "enum": ["create", "update", "mark_step", "finish"],
            "description": (
                "The command to execute. Available commands: "
                "create, update, mark_step, finish"
            ),
        }

    def _get_title_property(self) -> Dict[str, Any]:
        return {
            "type": "string",
            "description": (
                "Title for the plan. Required for create command, "
                "optional for update command."
            ),
        }

    def _get_steps_property(self) -> Dict[str, Any]:
        return {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "List of plan steps. Required for create command, "
                "optional for update command."
            ),
        }

    def _get_step_index_property(self) -> Dict[str, Any]:
        return {
            "type": "integer",
            "description": (
                "Index of the step to update (0-based). "
                "Required for mark_step command."
            ),
        }

    def _get_step_status_property(self) -> Dict[str, Any]:
        return {
            "type": "string",
            "enum": ["not_started", "in_progress", "completed", "blocked"],
            "description": (
                "Status to set for a step. Used with mark_step command."
            ),
        }

    def _get_step_notes_property(self) -> Dict[str, Any]:
        return {
            "type": "string",
            "description": (
                "Additional notes for a step. Optional for mark_step command."
            ),
        }

    # ---------- 执行入口 ----------

    async def execute(self, input: Any):
        if not isinstance(input, dict):
            raise ValueError("Input must be a Map")

        command = input.get("command")
        if not command:
            raise ValueError("Command is required")

        handler = self.command_handlers.get(command)
        if not handler:
            raise ValueError(f"Unknown command: {command}")

        return handler(input)

    # ---------- Command 实现 ----------

    def create_plan(self, params: Dict[str, Any]) -> str:
        title = params.get("title")
        steps = params.get("steps")

        if title is None or steps is None:
            raise ValueError(
                "创建计划失败：缺少必填参数 title 或 steps")

        if self.plan is not None:
            raise RuntimeError(
                "当前已存在计划，如需重新创建请先清除原计划"
            )

        self.plan = Planner.create(title, steps)
        return "我已创建plan"

    def update_plan(self, params: Dict[str, Any]) -> str:
        if self.plan is None:
            raise RuntimeError("当前尚未创建计划，请先创建计划")

        title = params.get("title")
        steps = params.get("steps")
        self.plan.update(title, steps)

        return "我已更新plan"

    def mark_step(self, params: Dict[str, Any]) -> str:
        if self.plan is None:
            raise RuntimeError("当前尚未创建计划，请先创建计划")

        step_index = params.get("step_index")
        step_status = params.get("step_status")
        step_notes = params.get("step_notes")

        if step_index is None:
            raise ValueError("mark_step 操作必须提供 step_index")

        self.plan.update_step_status(step_index, step_status, step_notes)
        return f"我已标记plan {step_index}:{self.plan.steps[step_index]} 为 {step_status}"

    def finish_plan(self, params: Dict[str, Any]) -> str:
        if self.plan is None:
            self.plan = Planner()
        else:
            for idx in range(len(self.plan.steps)):
                self.plan.update_step_status(idx, "completed", "")

        return "我已更新plan为完成状态"

    # ---------- 其他方法 ----------

    def step_plan(self):
        if self.plan:
            self.plan.step_plan()

    def get_format_plan(self) -> str:
        if self.plan is None:
            return "目前还没有Plan"
        return self.plan.format()
