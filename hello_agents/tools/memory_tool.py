"""记忆工具

为 hello-agents-learn 框架提供**跨会话持久化记忆**能力。
可以作为工具添加到任何 Agent 中，让 Agent 具备“长期记忆”功能。

职责与特性（与 `agent_memory.Memory` 区分）：

- **Memory（agent_memory）**：以 `agent_id/request_id` 为粒度，单次执行期的短期对话上下文，仅存在于内存中，不做持久化。
- **MemoryTool（本文件）**：以 `user_id` 为隔离标识，负责**跨会话、可持久化、可检索、可衰减**的长期记忆管理。

实现说明：

- 当前实现采用 **本地 JSON 文件** 作为简单的持久化后端：
  - 存储路径：`<项目根>/memory_storage/{user_id}.json`
  - 结构：`[{"id": "...", "content": "...", "memory_type": "...", "importance": 0.5, "timestamp": "...", "metadata": {...}}, ...]`
- 对外暴露统一的工具接口：`execute(input: Any) -> Any`，兼容 LLM 的工具调用规范。
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from hello_agents.core.tool.base_tool import BaseTool


class MemoryTool(BaseTool):
    """
    记忆工具 - 提供可持久化、可索引、可衰减的记忆管理。

    用途：
    - 由 Agent 在合适的时机（如：任务结束、每 N 轮对话）写入重要记忆
    - 由上下文管理器（ContextManager / ContextBuilder）在构建上下文时检索相关记忆
    """

    name = "memory"
    description = (
        "对记忆进行可持久化、可索引、可衰减的状态管理，用于解决模型对话状态的遗忘"
        "(每一次 API 调用都是一次独立、无关联的计算)"
    )

    def __init__(
        self,
        user_id: str = "default_user",
        memory_types: Optional[List[str]] = None,
        storage_dir: Optional[str] = None,
    ):
        """
        Args:
            user_id: 记忆隔离标识（通常为用户 ID / ERP 等）
            memory_types: 支持的记忆类型列表
            storage_dir: 持久化目录，默认使用项目根目录下的 memory_storage
        """
        self.user_id = user_id
        self.memory_types = memory_types or ["working", "episodic", "semantic", "perceptual"]

        # 计算默认存储目录：<project_root>/memory_storage
        if storage_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            storage_dir = os.path.join(project_root, "memory_storage")
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self._memories: List[Dict[str, Any]] = self._load_memories()

        # 会话状态（用于 auto_record_conversation 等便捷方法）
        self.current_session_id: Optional[str] = None
        self.conversation_count: int = 0

    # ------------------------------------------------------------------
    # BaseTool 标准接口
    # ------------------------------------------------------------------
    def execute(self, input: Any) -> Any:
        """
        执行工具（BaseTool 要求的接口）
        
        Args:
            input: 可以是字典（包含action等参数）或字符串（action名称）
            
        Returns:
            执行结果
        """
        # 兼容不同的调用方式
        if isinstance(input, str):
            # 如果只传入action字符串，返回提示
            return f"❌ 请提供完整的参数字典，包含action字段。当前传入: {input}"
        elif isinstance(input, dict):
            return self.run(input)
        else:
            return f"❌ 不支持的输入类型: {type(input)}"

    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具（非展开模式）

        Args:
            parameters: 工具参数字典，必须包含action参数

        Returns:
            执行结果字符串
        """

        if not self._validate_parameters(parameters):
            return "❌ 参数验证失败：缺少必需的参数"

        action = parameters.get("action")

        # 根据action调用对应的方法，传入提取的参数
        if action == "add":
            return self._add_memory(
                content=parameters.get("content", ""),
                memory_type=parameters.get("memory_type", "working"),
                importance=parameters.get("importance", 0.5),
                file_path=parameters.get("file_path"),
                modality=parameters.get("modality")
            )
        elif action == "search":
            return self._search_memory(
                query=parameters.get("query"),
                limit=parameters.get("limit", 5),
                memory_type=parameters.get("memory_type"),
                min_importance=parameters.get("min_importance", 0.1)
            )
        elif action == "summary":
            return self._get_summary(limit=parameters.get("limit", 10))
        elif action == "stats":
            return self._get_stats()
        elif action == "update":
            return self._update_memory(
                memory_id=parameters.get("memory_id"),
                content=parameters.get("content"),
                importance=parameters.get("importance")
            )
        elif action == "remove":
            return self._remove_memory(memory_id=parameters.get("memory_id"))
        elif action == "forget":
            return self._forget(
                strategy=parameters.get("strategy", "importance_based"),
                threshold=parameters.get("threshold", 0.1),
                max_age_days=parameters.get("max_age_days", 30)
            )
        elif action == "consolidate":
            return self._consolidate(
                from_type=parameters.get("from_type", "working"),
                to_type=parameters.get("to_type", "episodic"),
                importance_threshold=parameters.get("importance_threshold", 0.7)
            )
        elif action == "clear_all":
            return self._clear_all()
        else:
            return f"❌ 不支持的操作: {action}"

    def to_params(self) -> Dict[str, Any]:
        """返回工具参数定义 - BaseTool 要求的接口"""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "要执行的操作："
                        "add(添加记忆), search(搜索记忆), summary(获取摘要), stats(获取统计), "
                        "update(更新记忆), remove(删除记忆), forget(遗忘记忆), consolidate(整合记忆), clear_all(清空所有记忆)"
                    ),
                    "enum": ["add", "search", "summary", "stats", "update", "remove", "forget", "consolidate", "clear_all"]
                },
                "content": {"type": "string", "description": "记忆内容（add/update时可用；感知记忆可作描述）"},
                "query": {"type": "string", "description": "搜索查询（search时可用）"},
                "memory_type": {"type": "string", "description": "记忆类型：working, episodic, semantic, perceptual（默认：working）", "enum": ["working", "episodic", "semantic", "perceptual"]},
                "importance": {"type": "number", "description": "重要性分数，0.0-1.0（add/update时可用）"},
                "limit": {"type": "integer", "description": "搜索结果数量限制（默认：5）"},
                "memory_id": {"type": "string", "description": "目标记忆ID（update/remove时必需）"},
                "file_path": {"type": "string", "description": "感知记忆：本地文件路径（image/audio）"},
                "modality": {"type": "string", "description": "感知记忆模态：text/image/audio（不传则按扩展名推断）"},
                "strategy": {"type": "string", "description": "遗忘策略：importance_based/time_based/capacity_based（forget时可用）", "enum": ["importance_based", "time_based", "capacity_based"]},
                "threshold": {"type": "number", "description": "遗忘阈值（forget时可用，默认0.1）"},
                "max_age_days": {"type": "integer", "description": "最大保留天数（forget策略为time_based时可用）"},
                "from_type": {"type": "string", "description": "整合来源类型（consolidate时可用，默认working）"},
                "to_type": {"type": "string", "description": "整合目标类型（consolidate时可用，默认episodic）"},
                "importance_threshold": {"type": "number", "description": "整合重要性阈值（默认0.7）"},
            },
            "required": ["action"]
        }

    # ------------------------------------------------------------------
    # 内部工具方法：参数校验 & 持久化
    # ------------------------------------------------------------------
    def _validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """最小化参数校验（只校验 action 必填）"""
        return isinstance(parameters, dict) and "action" in parameters

    @property
    def _storage_path(self) -> str:
        """当前 user_id 对应的持久化文件路径"""
        filename = f"{self.user_id}.json"
        return os.path.join(self.storage_dir, filename)

    def _load_memories(self) -> List[Dict[str, Any]]:
        """从本地 JSON 文件加载记忆列表"""
        try:
            if not os.path.exists(self._storage_path):
                return []
            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return []
        except Exception:
            # 任何异常均回退为空列表，避免影响 Agent 运行
            return []

    def _save_memories(self) -> None:
        """将当前记忆列表持久化到本地"""
        try:
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(self._memories, f, ensure_ascii=False, indent=2)
        except Exception:
            # 持久化失败不抛出异常，避免影响主流程
            pass

    # ------------------------------------------------------------------
    # 记忆增删改查 & 策略
    # ------------------------------------------------------------------
    def _add_memory(
        self,
        content: str = "",
        memory_type: str = "working",
        importance: float = 0.5,
        file_path: Optional[str] = None,
        modality: Optional[str] = None,
        **extra_metadata: Any,
    ) -> str:
        """添加记忆

        Args:
            content: 记忆内容
            memory_type: 记忆类型：working(工作记忆), episodic(情景记忆), semantic(语义记忆), perceptual(感知记忆)
            importance: 重要性分数，0.0-1.0
            file_path: 感知记忆：本地文件路径（image/audio）
            modality: 感知记忆模态：text/image/audio（不传则按扩展名推断）
            extra_metadata: 其它透传到 metadata 的字段（如 type、conversation_id 等）

        Returns:
            执行结果
        """
        try:
            if memory_type not in self.memory_types:
                return f"❌ 不支持的记忆类型: {memory_type}"

            # 确保会话 ID 存在（用于统计与追踪）
            if self.current_session_id is None:
                self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            metadata: Dict[str, Any] = dict(extra_metadata or {})

            # 感知记忆文件支持：注入 raw_data 与模态信息
            if memory_type == "perceptual" and file_path:
                inferred = modality or self._infer_modality(file_path)
                metadata.setdefault("modality", inferred)
                metadata.setdefault("raw_data", file_path)

            # 添加用户和会话信息到元数据（记忆与 user_id 相关，可跨会话）
            metadata.update(
                {
                    "user_id": self.user_id,
                    "session_id": self.current_session_id,
                }
            )

            memory_id = str(uuid.uuid4())
            now_str = datetime.now().isoformat()

            record = {
                "id": memory_id,
                "content": content,
                "memory_type": memory_type,
                "user_id": self.user_id,
                "timestamp": now_str,
                "importance": float(importance),
                "metadata": metadata,
            }
            self._memories.append(record)
            self._save_memories()

            return f"✅ 记忆已添加 (ID: {memory_id[:8]}...)"

        except Exception as e:
            return f"❌ 添加记忆失败: {str(e)}"

    def _infer_modality(self, path: str) -> str:
        """根据扩展名推断模态（默认image/audio/text）"""
        try:
            ext = (path.rsplit('.', 1)[-1] or '').lower()
            if ext in {"png", "jpg", "jpeg", "bmp", "gif", "webp"}:
                return "image"
            if ext in {"mp3", "wav", "flac", "m4a", "ogg"}:
                return "audio"
            return "text"
        except Exception:
            return "text"

    def _search_memory(
        self,
        query: str,
        limit: int = 5,
        memory_type: str = None,
        min_importance: float = 0.1
    ) -> str:
        """搜索记忆

        Args:
            query: 搜索查询内容
            limit: 搜索结果数量限制
            memory_type: 限定记忆类型：working/episodic/semantic/perceptual
            min_importance: 最低重要性阈值

        Returns:
            搜索结果
        """
        try:
            results: List[Dict[str, Any]] = []
            q = (query or "").lower()
            for m in self._memories:
                if m.get("importance", 0.0) < float(min_importance):
                    continue
                if memory_type and m.get("memory_type") != memory_type:
                    continue
                if q and q not in (m.get("content") or "").lower():
                    continue
                results.append(m)

            results = results[: max(0, int(limit))]

            if not results:
                return f"🔍 未找到与 '{query}' 相关的记忆"

            # 格式化结果
            formatted_results: List[str] = []
            formatted_results.append(f"🔍 找到 {len(results)} 条相关记忆:")

            for i, memory in enumerate(results, 1):
                memory_type_label = {
                    "working": "工作记忆",
                    "episodic": "情景记忆",
                    "semantic": "语义记忆",
                    "perceptual": "感知记忆",
                }.get(memory.get("memory_type", "working"), memory.get("memory_type", "working"))

                content_str = memory.get("content", "") or ""
                content_preview = content_str[:80] + "..." if len(content_str) > 80 else content_str
                formatted_results.append(
                    f"{i}. [{memory_type_label}] {content_preview} (重要性: {memory.get('importance', 0):.2f})"
                )

            return "\n".join(formatted_results)

        except Exception as e:
            return f"❌ 搜索记忆失败: {str(e)}"

    def _get_summary(self, limit: int = 10) -> str:
        """获取记忆摘要

        Args:
            limit: 显示的重要记忆数量

        Returns:
            记忆摘要
        """
        try:
            total = len(self._memories)
            summary_parts: List[str] = [
                "📊 记忆系统摘要",
                f"总记忆数: {total}",
                f"当前会话: {self.current_session_id or '未开始'}",
                f"对话轮次: {self.conversation_count}",
            ]

            # 各类型记忆统计
            if total > 0:
                by_type: Dict[str, Dict[str, Any]] = {}
                for m in self._memories:
                    t = m.get("memory_type", "working")
                    info = by_type.setdefault(t, {"count": 0, "sum_importance": 0.0})
                    info["count"] += 1
                    info["sum_importance"] += float(m.get("importance", 0.0))

                summary_parts.append("\n📋 记忆类型分布:")
                for memory_type, info in by_type.items():
                    count = info["count"]
                    avg_importance = info["sum_importance"] / max(count, 1)
                    type_label = {
                        "working": "工作记忆",
                        "episodic": "情景记忆",
                        "semantic": "语义记忆",
                        "perceptual": "感知记忆",
                    }.get(memory_type, memory_type)
                    summary_parts.append(
                        f"  • {type_label}: {count} 条 (平均重要性: {avg_importance:.2f})"
                    )

            # 重要记忆（按 importance 排序，取前 N 条）
            important_memories = sorted(
                self._memories,
                key=lambda m: float(m.get("importance", 0.0)),
                reverse=True,
            )
            important_memories = important_memories[: max(0, int(limit))]

            if important_memories:
                summary_parts.append(f"\n⭐ 重要记忆 (前{len(important_memories)}条):")
                for i, memory in enumerate(important_memories, 1):
                    content = memory.get("content", "") or ""
                    content_preview = content[:60] + "..." if len(content) > 60 else content
                    summary_parts.append(
                        f"  {i}. {content_preview} (重要性: {memory.get('importance', 0):.2f})"
                    )

            return "\n".join(summary_parts)

        except Exception as e:
            return f"❌ 获取摘要失败: {str(e)}"

    def _get_stats(self) -> str:
        """获取统计信息

        Returns:
            统计信息
        """
        try:
            total = len(self._memories)
            stats_info = [
                "📈 记忆系统统计",
                f"总记忆数: {total}",
                f"启用的记忆类型: {', '.join(self.memory_types)}",
                f"会话ID: {self.current_session_id or '未开始'}",
                f"对话轮次: {self.conversation_count}",
            ]
            return "\n".join(stats_info)
        except Exception as e:
            return f"❌ 获取统计信息失败: {str(e)}"

    def auto_record_conversation(self, user_input: str, agent_response: str):
        """自动记录对话

        这个方法可以被 Agent 调用来自动记录对话历史
        """
        self.conversation_count += 1

        # 记录用户输入
        self._add_memory(
            content=f"用户: {user_input}",
            memory_type="working",
            importance=0.6,
            type="user_input",
            conversation_id=self.conversation_count,
        )

        # 记录 Agent 响应
        self._add_memory(
            content=f"助手: {agent_response}",
            memory_type="working",
            importance=0.7,
            type="agent_response",
            conversation_id=self.conversation_count,
        )

        # 如果是重要对话，记录为情景记忆
        if len(agent_response) > 100 or "重要" in user_input or "记住" in user_input:
            interaction_content = f"对话 - 用户: {user_input}\n助手: {agent_response}"
            self._add_memory(
                content=interaction_content,
                memory_type="episodic",
                importance=0.8,
                type="interaction",
                conversation_id=self.conversation_count,
            )

    def _update_memory(self, memory_id: str, content: str = None, importance: float = None) -> str:
        """更新记忆

        Args:
            memory_id: 要更新的记忆ID
            content: 新的记忆内容
            importance: 新的重要性分数

        Returns:
            执行结果
        """
        try:
            if not memory_id:
                return "❌ 更新记忆失败: 缺少 memory_id"

            updated = False
            for m in self._memories:
                if m.get("id") == memory_id:
                    if content is not None:
                        m["content"] = content
                    if importance is not None:
                        m["importance"] = float(importance)
                    updated = True
                    break

            if updated:
                self._save_memories()
                return "✅ 记忆已更新"
            return "⚠️ 未找到要更新的记忆"

        except Exception as e:
            return f"❌ 更新记忆失败: {str(e)}"

    def _remove_memory(self, memory_id: str) -> str:
        """删除记忆

        Args:
            memory_id: 要删除的记忆ID

        Returns:
            执行结果
        """
        try:
            before = len(self._memories)
            self._memories = [m for m in self._memories if m.get("id") != memory_id]
            after = len(self._memories)
            self._save_memories()
            return "✅ 记忆已删除" if after < before else "⚠️ 未找到要删除的记忆"
        except Exception as e:
            return f"❌ 删除记忆失败: {str(e)}"

    def _forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 30) -> str:
        """遗忘记忆（支持多种策略）

        Args:
            strategy: 遗忘策略：importance_based(基于重要性)/time_based(基于时间)/capacity_based(基于容量)
            threshold: 遗忘阈值（importance_based时使用）
            max_age_days: 最大保留天数（time_based时使用）

        Returns:
            执行结果
        """
        try:
            before = len(self._memories)
            now = datetime.now()
            remaining: List[Dict[str, Any]] = []

            for m in self._memories:
                importance_val = float(m.get("importance", 0.0))
                ts_str = m.get("timestamp")
                try:
                    ts = datetime.fromisoformat(ts_str) if ts_str else now
                except Exception:
                    ts = now
                age_days = (now - ts).days

                keep = True
                if strategy == "importance_based" and importance_val < float(threshold):
                    keep = False
                elif strategy == "time_based" and age_days > int(max_age_days):
                    keep = False
                elif strategy == "capacity_based":
                    # 简单实现：超过阈值则按重要性从低到高丢弃
                    # 这里 threshold 被解释为“最大保留条数比例”，例如 0.8 表示只保留 80% 最新/重要的
                    keep = True  # 先全部保留，后面统一处理

                if keep:
                    remaining.append(m)

            # capacity_based 的二次处理
            if strategy == "capacity_based" and remaining:
                max_count = int(len(remaining) * float(threshold))
                if max_count <= 0:
                    remaining = []
                else:
                    remaining = sorted(
                        remaining,
                        key=lambda x: float(x.get("importance", 0.0)),
                        reverse=True,
                    )[:max_count]

            self._memories = remaining
            self._save_memories()

            removed = before - len(self._memories)
            return f"🧹 已遗忘 {removed} 条记忆（策略: {strategy}）"
        except Exception as e:
            return f"❌ 遗忘记忆失败: {str(e)}"

    def _consolidate(self, from_type: str = "working", to_type: str = "episodic", importance_threshold: float = 0.7) -> str:
        """整合记忆（将重要的短期记忆提升为长期记忆）

        Args:
            from_type: 来源记忆类型
            to_type: 目标记忆类型
            importance_threshold: 整合的重要性阈值

        Returns:
            执行结果
        """
        try:
            count = 0
            for m in self._memories:
                if (
                    m.get("memory_type") == from_type
                    and float(m.get("importance", 0.0)) >= float(importance_threshold)
                ):
                    m["memory_type"] = to_type
                    count += 1

            if count > 0:
                self._save_memories()
            return f"🔄 已整合 {count} 条记忆为长期记忆（{from_type} → {to_type}，阈值={importance_threshold}）"
        except Exception as e:
            return f"❌ 整合记忆失败: {str(e)}"

    def _clear_all(self) -> str:
        """清空所有记忆

        Returns:
            执行结果
        """
        try:
            count = len(self._memories)
            self._memories = []
            self._save_memories()
            return f"🧽 已清空所有记忆，共 {count} 条"
        except Exception as e:
            return f"❌ 清空记忆失败: {str(e)}"

    def add_knowledge(self, content: str, importance: float = 0.9):
        """添加知识到语义记忆

        便捷方法，用于添加重要知识
        """
        return self._add_memory(
            content=content,
            memory_type="semantic",
            importance=importance,
            knowledge_type="factual",
            source="manual",
        )

    def get_context_for_query(self, query: str, limit: int = 3) -> str:
        """为查询获取相关上下文

        这个方法可以被 Agent 调用来获取相关的记忆上下文
        """
        try:
            q = (query or "").lower()
            results: List[Dict[str, Any]] = []
            for m in self._memories:
                if float(m.get("importance", 0.0)) < 0.3:
                    continue
                if q and q not in (m.get("content") or "").lower():
                    continue
                results.append(m)

            if not results:
                return ""

            results = sorted(
                results,
                key=lambda x: float(x.get("importance", 0.0)),
                reverse=True,
            )[: max(0, int(limit))]

            context_parts = ["相关记忆:"]
            for memory in results:
                context_parts.append(f"- {memory.get('content', '')}")

            return "\n".join(context_parts)
        except Exception:
            return ""

    def clear_session(self):
        """清除当前会话（不会清空长期记忆，只重置会话计数）"""
        self.current_session_id = None
        self.conversation_count = 0

    def consolidate_memories(self):
        """整合记忆（便捷方法，等价于调用 _consolidate 默认参数）"""
        return self._consolidate()

    def forget_old_memories(self, max_age_days: int = 30):
        """遗忘旧记忆（便捷方法，基于时间窗口）"""
        return self._forget(strategy="time_based", max_age_days=max_age_days)
