import ast
import json
import os
import traceback
from typing import List, Optional

from hello_agents.agents_design.agent_prompt import REFLECTION_INITIAL, REFLECTION_NEXT_STEP_PROMPT, REFLECTION_SYSTEM_PROMPT, Digital_Employee_Prompt
from hello_agents.core.agent.agent_schema import AgentState
from hello_agents.core.agent.base_agent import BaseAgent
from hello_agents.core.llm.llm_schema import RoleType
from hello_agents.core.tool.file_util import FileUtil
from hello_agents.core.llm.message import Message
from hello_agents.core.tool.reflection_tool import ReflectionTool


class ReflectionAgent(BaseAgent):
    """
    Reflection Agent
    """

    def __init__(self, context, llm):
        super().__init__(
            name="Reflection",
            description="基于初始结果进行反思优化",
            system_prompt="",
            next_step_prompt="",
            llm=llm,
            context=context,
            max_steps=int(os.getenv("React_Max_Steps", 20))
        )
        self.reflection_tool = ReflectionTool(agent_context=context)
        self.available_tools.add_tool(self.reflection_tool)
        self.tool_calls: List[dict] = []
        self.max_observe: Optional[int] = None

        # ---------- Prompt ----------
        self.system_prompt = REFLECTION_SYSTEM_PROMPT.format(
            basePrompt=context.base_prompt,
            query=context.query,
            date=context.date_info
        )

        self.next_step_prompt = REFLECTION_NEXT_STEP_PROMPT.format(
            query=context.query)

        # ---------- Snapshot(快照-保存原始版本) ----------
        self.system_prompt_snapshot = self.system_prompt
        self.next_step_prompt_snapshot = self.next_step_prompt

        # ---------- Runtime ----------
        self.digital_employee_prompt = Digital_Employee_Prompt

        self.last_attempt = None  # 上一轮回答

    # ======================================================================
    # step =一次 think + act
    # ======================================================================
    async def step(self) -> str:
        # 判断memory是否只有一条消息如果是就初始执行，如果不是就反思优化
        if len(self.memory.messages) == 1:
            self.task = self.memory.messages[0].content
            self.last_attempt = await self.initial_task()
            return self.last_attempt
        else:
            should_continue = await self.think()
            if not should_continue:
                self.state = AgentState.FINISHED
                return self.memory.get_last_message().content
            return await self.act()

    async def initial_task(self):
        """
        一次完成任务
        """
        last_msg = self.memory.get_last_message()
        if last_msg is not None and last_msg.role != RoleType.USER:
            raise
        system_prompt = REFLECTION_INITIAL
        response = await self.llm.ask_llm_once(
            context=self.context,
            messages=self.memory.messages,
            system_msgs=Message.system_message(system_prompt)
        )
        msg = Message.assistant_message(response)
        self.memory.add_message(msg)
        return "初始结果："+response

    # ======================================================================
    # THINK
    # ======================================================================
    async def think(self) -> bool:
        """
        使用 LLM 进行思考，决定下一步行动：是否应该优化
        """
        try:
            # ---------- 注入 files ----------
            files_str = FileUtil.format_file_info(
                self.context.product_files, filter_internal_file=True
            )

            self.system_prompt = self.system_prompt_snapshot.replace(
                "{files}", files_str
            )
            # ----------如果user message不是最后一个消息，构建一个推进的提示词 ----------
            last_msg = self.memory.get_last_message()
            if last_msg is not None and last_msg.role != RoleType.USER:
                include_next_step_prompt = True
            else:
                include_next_step_prompt = False

            self.context.stream_message_type = "tool_thought"

            self.system_prompt = self.system_prompt_snapshot.replace(
                "{content}", self.last_attempt
            )
            # ---------- ask tool ----------
            response = await self.llm.ask_tool(
                context=self.context,
                messages=self.memory.messages,
                system_msgs=Message.system_message(self.system_prompt),
                tools=self.available_tools,
                tool_choice=self.context.tool_choice,
                include_next_step_prompt=include_next_step_prompt,
                next_step_prompt=self.system_prompt
            )

            self.tool_calls = response.tool_calls or []

            # ---------- 输出 thought ----------
            if not self.context.is_stream and response.content:
                self.context.printer.send(
                    f"tool_thought:{response.content}", "tool_thought")

            # ---------- assistant message ----------
            if self.tool_calls and self.llm.function_call_type != "struct_parse":
                msg = Message.from_tool_calls(
                    response.content,
                    self.tool_calls
                )
            else:
                msg = Message.assistant_message(response.content)

            self.memory.add_message(msg)

            return True

        except Exception as e:
            traceback.print_exc()
            self.update_memory(
                RoleType.ASSISTANT,
                f"Error encountered while processing: {e}"
            )
            self.state = AgentState.FINISHED
            return False

    # ======================================================================
    # ACT
    # ======================================================================
    async def act(self) -> str:
        if not self.tool_calls:
            self.state = AgentState.FINISHED
            return self.last_attempt

        if self.tool_calls[0].function.name.lower() == "finish":
            self.state = AgentState.FINISHED
            self.update_memory(
                RoleType.ASSISTANT,
                content=self.tool_calls[0].function.arguments
            )
            return self.last_attempt

        tool_results = await self.execute_tools(self.tool_calls)

        results = []

        for call in self.tool_calls:
            tool_id = call.id

            result = tool_results.get(tool_id, "")
            result_dict = ast.literal_eval(result)
            if result_dict["reflection_status"] == "no":
                self.state = AgentState.FINISHED
                return self.last_attempt
            else:
                next_step_prompt_snapshot_1 = self.next_step_prompt_snapshot
                next_step_prompt_snapshot_1 = next_step_prompt_snapshot_1.replace(
                    "{last_attempt}", self.last_attempt
                )
                next_step_prompt_snapshot_1 = next_step_prompt_snapshot_1.replace(
                    "{feedback}", result_dict["reflection_feedback"]
                )
                response = await self.llm.ask_llm_once(
                    context=self.context,
                    messages=self.memory.messages,
                    system_msgs=Message.system_message(
                        next_step_prompt_snapshot_1)
                )
                self.last_attempt = response

                self.update_memory(
                    RoleType.TOOL,
                    self.last_attempt,
                    None,
                    tool_id,
                )

                results.append(self.last_attempt)

        return "\n\n".join(results)
