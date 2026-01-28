import asyncio
import json
import re
import traceback
from typing import List, Optional
from pydantic import Field
from hello_agents.core.agent.agent_memory import Memory
from hello_agents.core.agent.agent_schema import AgentContext, AgentState
from hello_agents.core.agent.agent_contextmanager import ContextBuilder, ContextConfig
from hello_agents.core.llm.llm import LLMClient
from hello_agents.core.llm.llm_schema import RoleType, ToolCall
from hello_agents.core.llm.message import Message
from hello_agents.core.tool.tool_collection import ToolCollection
from hello_agents.tools.memory_tool import MemoryTool

# ===== BaseAgent =====


class BaseAgent:
    """Agent 基类"""

    def __init__(
        self,
        name: str = Field(description="Agent 的唯一名称，用于标识、日志记录与调度"),
        description: str = Field(description="Agent 的职责说明（人类可读，不直接参与推理）"),
        system_prompt: str = Field(
            description="系统级 Prompt,用于定义 Agent 的角色、能力边界与全局行为约束"),
        next_step_prompt: str = Field(
            description="单步推理引导 Prompt，用于驱动 Agent 决定“下一步做什么”"),
        llm: Optional[LLMClient] = Field(
            description="Agent 绑定的 LLM 实例，负责实际推理与文本生成"),
        context: Optional[AgentContext] = Field(
            description="Agent 运行上下文，用于保存状态、历史对话及中间结果"),
        max_steps: int = Field(
            default=10, description="Agent 允许执行的最大推理步数，用于防止无限循环"),
        duplicate_threshold: int = Field(
            default=2, description="连续重复输出或决策的阈值，用于检测 Agent 是否陷入死循环"),
        # ===== Memory & ContextManager =====
        memory_tool: Optional[MemoryTool] = Field(
            default=None,
            description="跨会话用户级长期记忆工具（MemoryTool），以 user_id 为隔离标识"),
        context_config: Optional[ContextConfig] = Field(
            default=None,
            description="GSSC 上下文构建配置（ContextConfig），用于控制 token 预算与筛选策略"),
    ):
        # core
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.next_step_prompt = next_step_prompt
        # 角色级 / 组织级配置  --应该从配置中来
        self.digital_employee_prompt: Optional[str] = None
        self.available_tools = ToolCollection()

        # ---- 短期记忆：单次 Agent 执行期上下文（基于 agent_id/request_id）----
        self.memory = Memory()

        # ---- 长期记忆工具：跨会话、按 user_id 隔离的 MemoryTool ----
        self.memory_tool = memory_tool

        # ---- 上下文管理：GSSC 流水线（Gather-Select-Structure-Compress）----
        # 仅在存在 Printer 时才启用，防止在无输出通道的情况下创建无用实例。
        self.context_builder: Optional[ContextBuilder] = None
        if context and getattr(context, "printer", None) is not None:
            self.context_builder = ContextBuilder(
                printer=context.printer,
                memory_tool=self.memory_tool,
                rag_tool=None,
                config=context_config or ContextConfig(),
            )

        # 用于跟踪 Memory 中的“非 system 消息”数量，驱动每 10 条进行一次记忆整合。
        self._non_system_msg_counter: int = 0
        self.llm = llm
        self.context = context

        # 执行控制
        self.state = AgentState.IDLE
        self.max_steps = max_steps
        self.current_step = 0
        self.duplicate_threshold = duplicate_threshold

    # ===== abstract step =====
    async def step(self):
        """
        执行单个 Agent 推理步骤
        """
        raise NotImplementedError

    # ===== main loop =====
    async def run(self, query: str):
        self.state = AgentState.IDLE
        self.current_step = 0

        if query:
            # 1) 写入当前 Agent 短期记忆（agent_memory）
            self.update_memory(RoleType.USER, query)

            # 2) 可选：写入用户级长期记忆（MemoryTool）
            #    这里记录的是“用户提出的原始问题”
            if self.memory_tool is not None:
                try:
                    self.memory_tool.execute(
                        {
                            "action": "add",
                            "content": f"[用户 Query] {query}",
                            "memory_type": "working",
                            "importance": 0.6,
                        }
                    )
                except Exception:
                    # 长期记忆失败不影响主流程
                    pass

        results: List[str] = []

        try:
            while self.current_step < self.max_steps and self.state != AgentState.FINISHED:
                self.current_step += 1
                req_id = self.context.request_id if self.context else "-"

                # 打印生命周期信息
                if self.context and self.context.printer:
                    self.context.printer.send(
                        f"{req_id} {self.name} Executing step {self.current_step}/{self.max_steps}"
                    )

                # 可选：在每一步开始前构建一次 GSSC 上下文（供子类按需使用）
                # 这里只负责触发构建，将结果缓存到实例属性，避免侵入各 Agent 具体实现。
                if self.context_builder is not None:
                    try:
                        # user_query 优先使用 AgentContext.query，其次回退为 run 入口的 query
                        user_query = (self.context.query if self.context else None) or query or ""
                        self._last_built_context: Optional[str] = self.context_builder.build(
                            user_query=user_query,
                            conversation_history=self.memory,
                        )
                    except Exception:
                        # 上下文构建失败不影响主流程
                        self._last_built_context = None

                step_result = await self.step()
                results.append(step_result)

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(
                    f"Terminated: Reached max steps ({self.max_steps})")

        except Exception:
            self.state = AgentState.ERROR
            traceback.print_exc()
            raise

        final_result = results[-1] if results else "No steps executed"

        # ---------- 单次 Agent 生命周期结束时：短期记忆 → 长期记忆 + 智能遗忘 ----------
        self._finalize_memory_lifecycle(final_result)

        return final_result

    # ===== memory =====
    def update_memory(
        self,
        role: RoleType,
        content: str,
        base64_image: Optional[str] = None,
        *args,
    ):
        if role == RoleType.USER:
            msg = Message.user_message(content, base64_image)
        elif role == RoleType.SYSTEM:
            msg = Message.system_message(content, base64_image)
        elif role == RoleType.ASSISTANT:
            msg = Message.assistant_message(content, base64_image)
        elif role == RoleType.TOOL:
            msg = Message.tool_message(content, args[0], base64_image)
        else:
            raise ValueError(f"Unsupported role type: {role}")

        # 1) 写入单次 Agent 的短期记忆（agent_memory）
        self.memory.add_message(msg)

        # 2) 统计非 system 消息条数，用于“每 10 条进行一次记忆整合”
        if role != RoleType.SYSTEM:
            self._non_system_msg_counter += 1
            self._maybe_consolidate_short_term_memory()

    # ===== tool execution =====
    async def execute_tool(self, command: ToolCall):
        try:
            func = command.function
            if not func or not func.name:
                return "Error: Invalid function call format"

            name = func.name
            args = json.loads(func.arguments or "{}")

            result = await self.available_tools.execute(name, args)
            req_id = self.context.request_id if self.context else "-"
            self.context.printer.send(
                f"{req_id} execute tool: {name} {args} result {result}")

            return str(result) if result is not None else ""

        except Exception as e:
            req_id = self.context.request_id if self.context else "-"
            self.context.printer.send(
                f"{req_id} execute tool {name if 'name' in locals() else '-'} failed: {e}")
            return f"Tool {name if 'name' in locals() else ''} Error."

    async def execute_tools(self, commands: List[ToolCall]):
        """
        并发执行多个工具调用命令并返回执行结果
        :param commands: 工具调用命令列表
        :return: key 为 tool_call.id，value 为执行结果
        """

        async def _run_tool(tool_call: ToolCall):
            try:
                result = await self.execute_tool(tool_call)
                return tool_call.id, result
            except Exception as e:
                return tool_call.id, f"Tool Error: {e}"

        tasks = [asyncio.create_task(_run_tool(cmd)) for cmd in commands]

        results = await asyncio.gather(*tasks)

        return {tool_id: result for tool_id, result in results}

    async def generate_digital_employee(self, task: str):
        # 1. 参数检查
        if not task:
            return

        try:
            # 2. 构建系统 Prompt
            formatted_prompt = self._format_system_prompt(task)

            # 2.1 可选：为“生成数字员工”这一调用注入 GSSC 上下文
            #     这里将上下文追加到 system_prompt 末尾，让模型在角色设定时参考历史记忆。
            if self.context_builder is not None:
                try:
                    gssc_context = self.context_builder.build(
                        user_query=task,
                        conversation_history=self.memory,
                    )
                    if gssc_context:
                        formatted_prompt = f"{formatted_prompt}\n\n{gssc_context}"
                except Exception:
                    # 上下文构建失败不影响数字员工生成
                    pass
            user_message = Message.user_message(formatted_prompt)

            # 3. 调用 LLM
            llm_response = await self.llm.ask_llm_once(
                context=self.context,
                messages=[user_message],
                system_msgs=[],
            )

            self.context.printer.send(
                f"requestId: {self.context.request_id} "
                f"task:{task} "
                f"generateDigitalEmployee: {llm_response}"
            )

            # 4. 解析 JSON
            json_obj = self._parse_digital_employee(llm_response)
            if json_obj:
                self.context.printer.send(
                    f"requestId:{self.context.request_id} "
                    f"generateDigitalEmployee parsed: {json_obj}"
                )
                self.context.tool_collection.update_digital_employee(json_obj)
                self.context.tool_collection.set_current_task(task)

                # 更新可用工具
                self.available_tools = self.context.tool_collection
            else:
                self.context.printer.send(
                    f"requestId: {self.context.request_id} "
                    f"generateDigitalEmployee failed"
                )

        except Exception as e:
            self.context.printer.send(
                f"requestId: {self.context.request_id} "
                f"in generateDigitalEmployee failed: {e}"
            )

    # ------------------------------------------------------------------
    # Memory 生命周期管理：短期记忆整合 + 智能遗忘
    # ------------------------------------------------------------------
    def _maybe_consolidate_short_term_memory(self) -> None:
        """
        每当非 system 消息数量达到 10 的倍数时，将最近一段对话整合为一条
        “情景记忆（episodic）”写入 MemoryTool。

        说明：
            - 只在存在 MemoryTool 时执行；
            - 整合内容为最近 10 条非系统消息的简要 JSON 结构，便于后续回顾式检索。
        """
        if self.memory_tool is None:
            return
        if self._non_system_msg_counter <= 0 or self._non_system_msg_counter % 10 != 0:
            return

        try:
            recent_history = self.memory.get_non_system_prompt(top_n=10)
            if not recent_history:
                return
            content = json.dumps(
                {
                    "type": "episodic_segment",
                    "agent": self.name,
                    "step_range": f"{self._non_system_msg_counter-9}-{self._non_system_msg_counter}",
                    "messages": recent_history,
                },
                ensure_ascii=False,
            )
            self.memory_tool.execute(
                {
                    "action": "add",
                    "content": content,
                    "memory_type": "episodic",
                    "importance": 0.7,
                }
            )
        except Exception:
            # 记忆整合失败不影响主流程
            pass

    def _finalize_memory_lifecycle(self, final_result: str) -> None:
        """
        单次 Agent 运行结束时的记忆收尾逻辑：

        1. 将本轮任务的关键结论写入长期记忆（semantic）
        2. 对短期 Memory 执行“智能遗忘”：
           - 保留系统指令
           - 清理工具调用相关内容
           - 可选：仅保留最近若干轮对话
        """
        # 1) 将最终结果写入长期语义记忆
        if self.memory_tool is not None and final_result:
            try:
                summary_prefix = f"[Agent {self.name} 本轮任务总结]"
                self.memory_tool.execute(
                    {
                        "action": "add",
                        "content": f"{summary_prefix}\n{final_result}",
                        "memory_type": "semantic",
                        "importance": 0.9,
                    }
                )
            except Exception:
                pass

        # 2) 对 agent_memory 执行“智能遗忘”
        #    当前实现策略：
        #      - 清除工具执行痕迹（tool messages + 带 tool_calls 的 assistant）
        #      - 保留系统消息与最近最多 20 条非系统对话
        try:
            # 2.1 先清理工具相关上下文
            self.memory.clear_tool_context()

            # 2.2 再做一次“截断”：仅保留最近 20 条非系统消息 + 所有 system 消息
            non_system = [
                m for m in self.memory.messages if m.role != RoleType.SYSTEM
            ]
            system_msgs = [m for m in self.memory.messages if m.role == RoleType.SYSTEM]
            truncated_non_system = non_system[-20:] if len(non_system) > 20 else non_system
            self.memory.messages = system_msgs + truncated_non_system

        except Exception:
            # 避免因遗忘逻辑异常影响主流程
            pass

    def _parse_digital_employee(self, response: str):
        """
        支持：
         * 格式一：
         *      ```json
         *      {
         *          "file_tool": "市场洞察专员"
         *      }
         *      ```
         * 格式二：
         *      {
         *          "file_tool": "市场洞察专员"
         *      }
        """
        if not response:
            return None

        json_string = response

        match = re.search(r"```\\s*json([\\s\\S]+?)```", response)
        if match:
            json_string = match.group(1).strip()

        try:
            return json.loads(json_string)
        except Exception as e:
            self.context.printer.send(
                f"requestId: {self.context.request_id} "
                f"parseDigitalEmployee error: {e}"
            )
            return None

    def _format_system_prompt(self, task: str) -> str:
        digital_employee_prompt = self.digital_employee_prompt
        if not digital_employee_prompt:
            raise RuntimeError("Digital employee prompt is not configured")

        tool_desc = []
        for tool in self.context.tool_collection.tool_map.values():
            tool_desc.append(
                f"工具名：{tool.name} 工具描述：{tool.description}"
            )

        return (
            digital_employee_prompt
            .replace("{{task}}", task)
            .replace("{{ToolsDesc}}", "\n".join(tool_desc))
            .replace("{{query}}", self.context.query)
        )
