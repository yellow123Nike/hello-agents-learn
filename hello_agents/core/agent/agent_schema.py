from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Any

from hello_agents.core.tool.tool_collection import ToolCollection
if TYPE_CHECKING:
    from hello_agents.core.printer.printer import Printer

"""
agent 状态枚举
"""


class AgentState(Enum):
    IDLE = "IDLE"         # 空闲状态
    RUNNING = "RUNNING"   # 运行状态
    FINISHED = "FINISHED"  # 完成状态
    ERROR = "ERROR"       # 错误状态


@dataclass
class AgentContext:
    # ========= 基础追踪信息 =========
    request_id: str
    session_id: Optional[str] = None

    # ========= 用户 & 任务语义 =========
    query: Optional[str] = None           # 用户原始问题
    task: Optional[str] = None            # 当前 Agent 子任务
    agent_type: Optional[int] = None      # AgentType 枚举值

    # ========= 流式与输出控制 =========
    # Printer 实例（WS / Console）
    printer: Optional["Printer"] = None
    is_stream: bool = False
    stream_message_type: str = "llm"

    # ========= 工具与能力 =========
    tool_collection: Optional[ToolCollection] = None  # ToolCollection

    # ========= Prompt & 模板 =========
    sop_prompt: Optional[str] = None
    base_prompt: Optional[str] = None
    template_type: Optional[str] = None   # markdown / text / card

    # ========= 文件 & RAG =========
    product_files: List[Any] = field(default_factory=list)
    task_product_files: List[Any] = field(default_factory=list)

    # ========= 环境信息 =========
    date_info: Optional[str] = None


class AgentType(Enum):
    """agent类型"""
    COMPREHENSIVE = 1
    WORKFLOW = 2
    PLAN_SOLVE = 3
    ROUTER = 4
    REACT = 5

    @staticmethod
    def from_code(value: int) -> "AgentType":
        """等价于 Java 的 fromCode(int value)"""
        for agent_type in AgentType:
            if agent_type.value == value:
                return agent_type
        raise ValueError(f"Invalid AgentType code: {value}")


@dataclass
class AgentRequest:
    # ========= 请求级别 =========
    request_id: Optional[str] = None          # 请求唯一标识
    erp: Optional[str] = None                 # 用户/员工标识
    query: Optional[str] = None               # 用户原始问题
    agent_type: Optional[int] = None          # Agent 类型
    base_prompt: Optional[str] = None         # 基础 Prompt
    sop_prompt: Optional[str] = None          # SOP Prompt
    is_stream: Optional[bool] = None          # 是否流式输出
    messages: List["AgentRequest.Message"] = field(default_factory=list)
    # 交付物产出格式：html(网页模式）， docs(文档模式）， table(表格模式）
    output_style: Optional[str] = None

    # =============================
    # 内嵌模型：Message
    # =============================
    @dataclass
    class Message:
        # user / assistant / system / tool
        role: Optional[str] = None
        content: Optional[str] = None          # 消息内容
        command_code: Optional[str] = None     # 指令/命令码
        upload_file: List["FileInformation"] = field(default_factory=list)
        files: List["FileInformation"] = field(default_factory=list)


@dataclass
class FileInformation:
    # ========= 当前态文件信息（系统加工后） =========
    file_name: Optional[str] = None          # 当前系统内文件名
    file_desc: Optional[str] = None          # 文件业务描述
    oss_url: Optional[str] = None            # 对象存储内部地址
    domain_url: Optional[str] = None         # 对外可访问地址
    file_size: Optional[int] = None          # 文件大小（字节）
    file_type: Optional[str] = None          # 文件类型 / MIME / 扩展名

    # ========= 原始态文件信息（可回溯） =========
    origin_file_name: Optional[str] = None   # 原始文件名
    origin_file_url: Optional[str] = None    # 原始业务访问地址
    origin_oss_url: Optional[str] = None     # 原始对象存储地址
    origin_domain_url: Optional[str] = None  # 原始对外访问地址
