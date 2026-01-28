"""ContextBuilder - GSSC流水线实现

实现 Gather-Select-Structure-Compress 上下文构建流程：
1. Gather: 从多源收集候选信息（历史、记忆、RAG、工具结果）
2. Select: 基于优先级、相关性、多样性筛选
3. Structure: 组织成结构化上下文模板
4. Compress: 在预算内压缩与规范化
"""
from dataclasses import dataclass
from datetime import datetime
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

from hello_agents.core.agent.agent_memory import Memory
from hello_agents.core.printer.printer import Printer
from hello_agents.tools.memory_tool import MemoryTool
from hello_agents.tools.rag_tool import RagTool


@dataclass
class ContextPacket:
    """候选信息包。候选基本单元
    Attributes:
        content: 信息内容
        timestamp: 时间戳（如果未提供，自动使用当前时间）
        token_count: Token 数量
        relevance_score: 相关性分数(0.0-1.0)
        metadata: 可选的元数据
    """
    content: str
    timestamp: Optional[datetime] = None
    token_count: int = 0
    relevance_score: float = 0.5
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.metadata is None:
            self.metadata = {}
        # 自动补全时间戳：如果未提供，使用当前时间
        if self.timestamp is None:
            self.timestamp = datetime.now()
        # 确保相关性分数在有效范围内
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))
        # 自动计算 token 数量
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)


@dataclass
class ContextConfig:
    """上下文构建配置
    Attributes:
        max_tokens: 最大 token 数量
        reserve_ratio: 为系统指令预留的比例(0.0-1.0)
        min_relevance: 最低相关性阈值
        enable_compression: 是否启用压缩
        recency_weight: 新近性权重(0.0-1.0)
        relevance_weight: 相关性权重(0.0-1.0)
        top_n: 历史对话最近n条消息
    """
    max_tokens: int = 3000
    reserve_ratio: float = 0.2
    min_relevance: float = 0.1
    enable_compression: bool = True
    recency_weight: float = 0.3
    relevance_weight: float = 0.7
    top_n: int = 10

    def __post_init__(self):
        """验证配置参数"""
        assert 0.0 <= self.reserve_ratio <= 1.0, "reserve_ratio 必须在 [0, 1] 范围内"
        assert 0.0 <= self.min_relevance <= 1.0, "min_relevance 必须在 [0, 1] 范围内"
        assert abs(self.recency_weight + self.relevance_weight - 1.0) < 1e-6, \
            "recency_weight + relevance_weight 必须等于 1.0"

    def get_available_tokens(self) -> int:
        """
        计算在当前配置下，可用于上下文内容的 token 预算。

        说明：
            预留 reserve_ratio 比例给系统指令等“刚性”内容，
            剩余部分为上下文候选内容可用的 token 数。
        """
        reserved = int(self.max_tokens * self.reserve_ratio)
        return max(self.max_tokens - reserved, 0)

class ContextBuilder:
    """上下文构建器 - GSSC流水线
    用法示例：
    ```python
    builder = ContextBuilder(
        memory_tool=memory_tool,
        rag_tool=rag_tool,
        config=ContextConfig(max_tokens=8000)
    )
    context = builder.build(
        user_query="用户问题",
        conversation_history=[...],
        system_instructions="系统指令"
    )
    ```
    """

    def __init__(
        self,
        printer: "Printer",
        memory_tool: Optional[MemoryTool] = None,
        rag_tool: Optional[RagTool] = None,
        config: Optional[ContextConfig] = None
    ):
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.config = config or ContextConfig()
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self.printer = printer
    def build(
        self,
        user_query: str,
        conversation_history: Optional[Memory] = None,
        additional_packets: Optional[List[ContextPacket]] = None
    ) -> str:
        """构建完整上下文
        Args:
            user_query: 用户查询
            conversation_history: 对话历史(包含系统指令system_instructions)
            additional_packets: 额外的上下文包
        Returns:
            结构化上下文字符串
        """
        # 1. Gather: 收集候选信息
        packets = self._gather(
            user_query=user_query,
            conversation_history=conversation_history or [],
            additional_packets=additional_packets or []
        )
        # 2. Select: 筛选与排序
        selected_packets = self._select(packets, user_query)
        # 3. Structure: 组织成结构化模板
        structured_context = self._structure(
            selected_packets=selected_packets,
            user_query=user_query,
        )
        # 4. Compress: 压缩与规范化（如果超预算）
        final_context = self._compress(structured_context)

        return final_context

    def _gather(
        self,
        user_query: str,
            conversation_history: Memory,
        additional_packets: List[ContextPacket]
    ) -> List[ContextPacket]:
        """Gather: 收集候选信息"""
        packets: List[ContextPacket] = []

        # P0: 对话历史中的系统指令（如果存在）
        system_prompt = conversation_history.get_system_prompt()
        if system_prompt:
            packets.append(
                ContextPacket(
                    content=system_prompt,
                    metadata={"type": "system_prompt", "priority": "high"},
                    relevance_score=1.0,
                )
            )

        # P1: 从记忆中获取任务状态与关键结论
        if self.memory_tool:
            try:
                # 搜索任务状态相关记忆
                state_results = self.memory_tool.execute({
                    "action": "search",
                    "query": "(任务状态 OR 子目标 OR 结论 OR 阻塞)",
                    "min_importance": 0.7,
                    "limit": 5
                })
                if state_results and "未找到" not in state_results:
                    packets.append(ContextPacket(
                        content=state_results,
                        metadata={"type": "task_state", "importance": "high"}
                    ))

                # 搜索与当前查询相关的记忆
                related_results = self.memory_tool.execute({
                    "action": "search",
                    "query": user_query,
                    "limit": 5
                })
                if related_results and "未找到" not in related_results:
                    packets.append(ContextPacket(
                        content=related_results,
                        metadata={"type": "related_memory"}
                    ))
            except Exception as e:
                print(f"⚠️ 记忆检索失败: {e}")

        # P2: 从RAG中获取事实证据
        if self.rag_tool:
            try:
                rag_results = self.rag_tool.run({
                    "action": "search",
                    "query": user_query,
                    "limit": 5
                })
                if rag_results and "未找到" not in rag_results and "错误" not in rag_results:
                    packets.append(ContextPacket(
                        content=rag_results,
                        metadata={"type": "knowledge_base"}
                    ))
            except Exception as e:
                print(f"⚠️ RAG检索失败: {e}")

        # P3: 对话历史（辅助材料）
        if conversation_history and not conversation_history.is_empty():
            # 只保留最近 N 条
            recent_history = conversation_history.get_non_system_prompt(top_n=self.config.top_n)
            history_text = json.dumps(recent_history, ensure_ascii=False)
            packets.append(
                ContextPacket(
                    content=history_text,
                    metadata={"type": "history"},
                    relevance_score=0.6,  # 历史消息的基础相关性
                )
            )

        # 添加额外包
        packets.extend(additional_packets)

        return packets

    def _select(
        self,
        packets: List[ContextPacket],
        user_query: str
    ) -> List[ContextPacket]:
        """Select: 基于分数与预算的筛选"""

        # 0) 分离system_prompt和非system_prompt
        system_packets = [p for p in packets if p.metadata.get(
            "type") == "system_prompt"]
        other_packets = [p for p in packets if p.metadata.get(
            "type") != "system_prompt"]
        
        # 1) 计算system_prompt占用的token
        system_tokens = sum(p.token_count for p in system_packets)

        # 2). 为其他信息计算综合分数
        scored_packets: List[Tuple[float, ContextPacket]] = []
        for packet in other_packets:
            # 2.1) 计算相关性（关键词重叠）或者语义相似度
            query_tokens = set(user_query.lower().split())
            content_tokens = set(packet.content.lower().split())
            if len(query_tokens) > 0:
                overlap = len(query_tokens & content_tokens)
                packet.relevance_score = overlap / len(query_tokens)
            else:
                packet.relevance_score = 0.0

            # 2.2) 计算新近性（指数衰减）
            def recency_score(ts: datetime) -> float:
                delta = max((datetime.now() - ts).total_seconds(), 0)
                tau = 3600  # 1小时时间尺度，可暴露到配置
                return math.exp(-delta / tau)
            rec = recency_score(packet.timestamp)
            # 2.3) 计算复合分：0.7*相关性 + 0.3*新近性
            score = self.config.relevance_weight * packet.relevance_score + self.config.recency_weight * rec
            # 2.4) 过滤低于最小相关性阈值的信息
            if packet.relevance_score >= self.config.min_relevance:
                scored_packets.append((score, packet))
        # 3). 按分数降序排序
        scored_packets.sort(key=lambda x: x[0], reverse=True)
        # 4). 贪心选择:按分数从高到低填充,直到达到 token 上限
        selected = system_packets.copy()
        current_tokens = system_tokens

        token_budget = self.config.max_tokens
        for score, packet in scored_packets:
            if current_tokens + packet.token_count <= token_budget:
                selected.append(packet)
                current_tokens += packet.token_count
            else:
                # Token 预算已满,停止选择
                break

        if self.printer is not None:
            # 通过 Printer 输出简单调试信息，方便观察上下文构建情况
            self.printer.send_simple(
                message_type="context_builder",
                message=f"[ContextBuilder] 选择了 {len(selected)} 个信息包, 共 {current_tokens} tokens",
            )
        return selected

    def _structure(
        self,
        selected_packets: List[ContextPacket],
        user_query: str
    ) -> str:
        """Structure: 组织成结构化上下文模板"""
        sections = []

        # [Role & Policies] - 系统指令
        p0_packets = [p for p in selected_packets if p.metadata.get(
            "type") == "system_prompt"]
        if p0_packets:
            role_section = "[Role & Policies]\n"
            role_section += "\n".join([p.content for p in p0_packets])
            sections.append(role_section)

        # [Task] - 当前任务
        sections.append(f"[Task]\n用户问题：{user_query}")

        # [State] - 任务状态
        p1_packets = [p for p in selected_packets if p.metadata.get(
            "type") == "task_state"]
        if p1_packets:
            state_section = "[State]\n关键进展与未决问题：\n"
            state_section += "\n".join([p.content for p in p1_packets])
            sections.append(state_section)

        # [Evidence] - 事实证据
        p2_packets = [
            p for p in selected_packets
            if p.metadata.get("type") in {"related_memory", "knowledge_base", "retrieval"}
        ]
        if p2_packets:
            evidence_section = "[Evidence]\n事实与引用：\n"
            for p in p2_packets:
                evidence_section += f"\n{p.content}\n"
            sections.append(evidence_section)

        # [Context] - 辅助材料（历史等）
        p3_packets = [p for p in selected_packets if p.metadata.get(
            "type") == "history"]
        if p3_packets:
            context_section = "[Context]\n对话历史与背景：\n"
            context_section += "\n".join([p.content for p in p3_packets])
            sections.append(context_section)

        # [Output] - 输出约束
        output_section = """[Output]
                            请按以下格式回答：
                            1. 结论（简洁明确）
                            2. 依据（列出支撑证据及来源）
                            3. 风险与假设（如有）
                            4. 下一步行动建议（如适用）"""
        sections.append(output_section)

        return "\n\n".join(sections)

    def _compress(self, context: str) -> str:
        """Compress: 压缩与规范化"""
        if not self.config.enable_compression:
            return context

        current_tokens = count_tokens(context)
        available_tokens = self.config.get_available_tokens()

        if current_tokens <= available_tokens:
            return context

        # 简单截断策略（保留前 N 个 token）
        # 实际应用中可用LLM做高保真摘要
        print(f"⚠️ 上下文超预算 ({current_tokens} > {available_tokens})，执行截断")

        # 按段落截断，保留结构
        lines = context.split("\n")
        compressed_lines = []
        used_tokens = 0

        for line in lines:
            line_tokens = count_tokens(line)
            if used_tokens + line_tokens > available_tokens:
                break
            compressed_lines.append(line)
            used_tokens += line_tokens

        return "\n".join(compressed_lines)


def count_tokens(text: str) -> int:
    """计算文本token数（使用tiktoken）"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # 降级方案：粗略估算（1 token ≈ 4 字符）
        return len(text) // 4
