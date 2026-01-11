
import json
import os
import re
import traceback
import asyncio
from typing import List, Optional
from hello_agents.agents_design.agent_prompt import PLANSOLVE_NEXT_STEP_PROMPT, PLANSOLVE_SYSTEM_PROMPT
from hello_agents.agents_design.react_agent import ReactAgent
from hello_agents.core.agent.agent_schema import AgentContext, AgentState
from hello_agents.core.agent.base_agent import BaseAgent
from hello_agents.core.llm.llm_schema import RoleType
from hello_agents.core.llm.message import Message
from hello_agents.core.tool.file_util import FileUtil
from hello_agents.core.tool.planning_tool import PlanningTool
from hello_agents.core.tool.tool_schema import ToolCall


class PlanSolveAgent(BaseAgent):
    """
    - 基于 ReAct Agent 扩展的 PlanSolve Agent
    - Plan负责制定全局计划
    - Solve(ReactAgent)负责执行当前任务
    """

    def __init__(self, context: AgentContext, llm, is_close_update: bool = False, max_steps: int = int(os.getenv("React_Max_Steps", 20))):
        super().__init__(
            name="planning",
            description="An agent that creates and manages plans to solve tasks",
            system_prompt="",
            next_step_prompt="",
            llm=llm,
            context=context,
            max_steps=max_steps,
        )

        self.tool_calls: List[ToolCall] = []
        self.max_observe: Optional[int] = None
        # 关闭动态更新 Plan 的开关 =1
        self.is_close_update: bool = is_close_update
        # PlanningTool
        self.planning_tool = PlanningTool(agent_context=context)
        self.available_tools.add_tool(self.planning_tool)

        self.system_prompt = PLANSOLVE_SYSTEM_PROMPT.format(
            sopPrompt=context.sop_prompt,
            date=context.date_info
        )
        self.next_step_prompt = PLANSOLVE_NEXT_STEP_PROMPT

        # ---------- Snapshot(快照-保存原始版本) ----------
        self.system_prompt_snapshot = self.system_prompt
        self.next_step_prompt_snapshot = self.next_step_prompt

    # ======================================================================
    # step =一次 think + act
    # ======================================================================

    async def step(self):
        should_continue = await self.think()
        if not should_continue:
            self.state = AgentState.FINISHED
            return self.memory.get_last_message().content
        return await self.act()

    # ======================================================================
    # THINK:生成或更新计划
    # ======================================================================
    async def think(self):
        """
        只负责：
        - prompt 构建（files 注入）
        - LLM tool planning（可能触发 PlanningTool）
        - memory 写 assistant/tool_call
        """
        try:
            # ---------- 注入 files ----------
            files_str = FileUtil.format_file_info(
                self.context.product_files, filter_internal_file=True
            )
            self.system_prompt = self.system_prompt_snapshot.replace(
                "{{files}}", files_str)
            self.next_step_prompt = self.next_step_prompt_snapshot.replace(
                "{{files}}", files_str)

            # ----------是否需要动态更新plan ----------
            if self.is_close_update and self.planning_tool.plan is not None:
                self.planning_tool.step_plan()
                return True
            # ---------- 确保 user message ----------
            last_msg = self.memory.get_last_message()
            if last_msg is not None and last_msg.role != RoleType.USER:
                include_next_step_prompt = True
            else:
                include_next_step_prompt = False

            self.context.stream_message_type = "plan_thought"

            # ---------- ask tool ----------
            response = await self.llm.ask_tool(
                context=self.context,
                messages=self.memory.messages,
                system_msgs=Message.system_message(self.system_prompt),
                tools=self.available_tools,
                tool_choice=self.context.tool_choice,
                include_next_step_prompt=include_next_step_prompt,
                next_step_prompt=self.next_step_prompt
            )

            self.tool_calls = response.tool_calls or []

            # ---------- 输出 thought ----------
            if not self.context.is_stream and response.content:
                self.context.printer.send(response.content, "plan_thought")

            # ---------- assistant message ----------
            if self.tool_calls and self.llm.function_call_type != "struct_parse":
                msg = Message.from_tool_calls(
                    response.content, self.tool_calls)
            else:
                msg = Message.assistant_message(response.content)

            self.memory.add_message(msg)
            return True

        except Exception as e:
            traceback.print_exc()
            self.update_memory(RoleType.ASSISTANT,
                               f"Error encountered while processing: {e}")
            self.state = AgentState.FINISHED
            return False

    # ======================================================================
    # ACT
    # ======================================================================
    async def act(self) -> str:
        """
        - 执行工具
        - 写 tool message
        - 如果形成 plan：返回下一步 task 或 finish
        """
        # ---------- 关闭动态更新 Plan：若已有 plan，直接输出 next task ----------
        if self.is_close_update and self.planning_tool.plan is not None:
            return await self.get_next_task()

        results: List[str] = []

        for call in self.tool_calls:
            tool_id = call.id
            result = await self.execute_tool(call)
            if self.max_observe:
                result = result[: self.max_observe]
            results.append(result)
            if self.llm.function_call_type == "struct_parse":
                last_msg = self.memory.get_last_message()
                last_msg.content += (
                    "\n工具执行结果为:\n" + result
                )
            else:
                self.update_memory(RoleType.TOOL, result, None, tool_id)

        # 工具执行后，如果形成了 plan：输出 next task / finish
        if self.planning_tool.plan is not None:
            if self.is_close_update:
                self.planning_tool.step_plan()
            return await self.get_next_task()

        return "\n\n".join(results)

    # ======================================================================
    # Next task
    # ======================================================================
    async def get_next_task(self) -> str:
        if self.planning_tool.plan is None:
            return ""

        # 获取当前计划
        plan = self.planning_tool.plan
        step_status = getattr(plan, "step_status", None) or getattr(
            plan, "stepStatus", None) or []
        current_step = plan.get_current_step() or ""
        # 检查是否所有步骤都已完成
        all_complete = True
        for status in step_status:
            if status != "completed":
                all_complete = False
                break
        # 如果所有步骤都完成，标记计划完成并返回
        if all_complete:
            self.state = AgentState.FINISHED
            self.context.printer.send(f"plan:{plan}", "plan")
            return f"finish:{plan}"
        # 如果有当前步骤，则执行它
        if current_step:
            self.context.printer.send(f"当前步骤:{current_step}", current_step)
            # 在这里调用 reactagent实现该步骤的任务
            context = AgentContext(
                request_id=f"{current_step}-context",
                base_prompt=self.context.base_prompt,
                date_info=self.context.date_info,
                printer=self.context.printer,
            )
            context.tool_collection = self.context.tool_collection
            agent = ReactAgent(
                context, self.llm)
            query = f"""
                # 原始问题:
                {self.memory.messages[0]}
                # 完整计划:
                {plan}
                steps--计划步骤列表
                step_status--计划执行情况
                steps_res--计划执行结果
                以上一一对应
                # 当前步骤:
                {current_step}
                请仅输出针对"当前步骤"的回答:
                
            """
            final_answer = await agent.run(query)
            final_answer = self.extract_after_last_finish(final_answer)
            self.planning_tool.plan.update_step_res(final_answer)
            self.context.printer.send(
                f"task:{current_step},result:{final_answer}", current_step)
            return str(current_step+":"+final_answer)

        return ""

    def extract_after_last_finish(self, text: str) -> str:
        """
        如果包含 Finish（忽略大小写），
        返回最后一个 Finish 之后的内容；
        否则返回原字符串。
        """
        matches = list(re.finditer(r'finish', text, re.IGNORECASE))
        if not matches:
            return text

        last_match = matches[-1]
        return text[last_match.end():].strip()
    # ======================================================================
    # run hook
    # ======================================================================

    def run(self, request: str) -> str:
        if self.planning_tool.plan is None:
            plan_pre_prompt = "分析问题并制定计划:"
            request = f"{plan_pre_prompt}{request}"
        return super().run(request)
