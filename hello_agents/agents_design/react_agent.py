
import json
import os
import traceback
from typing import List, Optional
from hello_agents.agents_design.agent_prompt import REACT_NEXT_STEP_PROMPT, REACT_SYSTEM_PROMPT, Digital_Employee_Prompt
from hello_agents.core.agent.agent_schema import AgentState
from hello_agents.core.agent.base_agent import BaseAgent
from hello_agents.core.llm.llm_schema import RoleType
from hello_agents.core.llm.message import Message
from hello_agents.core.tool.file_util import FileUtil


class ReactAgent(BaseAgent):
    """
    ReAct Agent
    """

    def __init__(self, context, llm):
        super().__init__(
            name="react",
            description="an agent that can execute tool calls.",
            system_prompt="",
            next_step_prompt="",
            llm=llm,
            context=context,
            max_steps=int(os.getenv("React_Max_Steps", 20))
        )
        self.available_tools = context.tool_collection
        self.tool_calls: List[dict] = []
        self.max_observe: Optional[int] = None

        # ---------- Prompt ----------
        self.system_prompt = REACT_SYSTEM_PROMPT.format(
            basePrompt=context.base_prompt,
            query=context.query,
            date=context.date_info
        )

        self.next_step_prompt = REACT_NEXT_STEP_PROMPT

        # ---------- Snapshot(快照-保存原始版本) ----------
        self.system_prompt_snapshot = self.system_prompt
        self.next_step_prompt_snapshot = self.next_step_prompt

        # ---------- Runtime ----------
        self.digital_employee_prompt = Digital_Employee_Prompt

    # ======================================================================
    # step =一次 think + act
    # ======================================================================
    async def step(self) -> str:
        should_continue = await self.think()
        if not should_continue:
            self.state = AgentState.FINISHED
            return self.memory.get_last_message().content
        return await self.act()

    # ======================================================================
    # THINK
    # ======================================================================
    async def think(self) -> bool:
        """
        使用 LLM 进行思考，决定下一步行动
        """

        try:
            # ---------- 注入 files ----------
            files_str = FileUtil.format_file_info(
                self.context.product_files, filter_internal_file=True
            )

            self.system_prompt = self.system_prompt_snapshot.replace(
                "{files}", files_str
            )
            self.next_step_prompt = self.next_step_prompt_snapshot.replace(
                "{files}", files_str
            )
            # ----------如果user message不是最后一个消息，构建一个推进的提示词 ----------
            last_msg = self.memory.get_last_message()
            if last_msg is not None and last_msg.role != RoleType.USER:
                include_next_step_prompt = True
            else:
                include_next_step_prompt = False

            self.context.stream_message_type = "tool_thought"

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
            return self.memory.get_last_message().content

        if self.tool_calls[0].function.name.lower() == "finish":
            self.state = AgentState.FINISHED
            self.update_memory(
                RoleType.ASSISTANT,
                content=self.tool_calls[0].function.arguments
            )
            return self.memory.get_last_message().content

        tool_results = await self.execute_tools(self.tool_calls)

        results = []

        for call in self.tool_calls:
            tool_id = call.id
            func = call.function
            tool_name = func.name
            args = json.loads(func.arguments or "{}")

            result = tool_results.get(tool_id, "")

            # ---------- 打印 tool result ----------
            if tool_name not in {
                "code_interpreter",
                "deep_search"
            }:
                if self.context.printer:
                    self.context.printer.send(
                        message_type="tool_result",
                        message={
                            "toolName": tool_name,
                            "toolParam": args,
                            "toolResult": result,
                        },
                    )

            if self.max_observe:
                result = result[: self.max_observe]

            # ---------- 写 memory ----------
            if self.llm.function_call_type == "struct_parse":
                last_msg = self.memory.get_last_message()
                last_msg.content += (
                    "\n工具执行结果为:\n" + result
                )
            else:
                self.update_memory(
                    RoleType.TOOL,
                    result,
                    None,
                    tool_id,
                )

            results.append(result)

        return "\n\n".join(results)
