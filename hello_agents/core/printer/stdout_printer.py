import json
import sys
from typing import Any, Optional
from hello_agents.core.printer.printer import Printer


class StdoutPrinter(Printer):
    def __init__(self, request):
        self.request = request
        self.agent_type: Optional[str] = None

    def send(
        self,
        message: Any,
        message_type: Optional[str] = "other",
        message_id: Optional[str] = None,
        digital_employee: Optional[str] = None,
        is_final: Optional[bool] = None,
    ):
        # deep_search 特殊序列化
        if message_type == "deep_search":
            try:
                message = json.dumps(message, ensure_ascii=False)
            except TypeError:
                message = str(message)

        prefix = f"[request_id={self.request.request_id}]"

        if self.agent_type:
            prefix += f"[agent_type={self.agent_type}]"

        if message_id:
            prefix += f"[msg_id={message_id}]"

        prefix += f"[type={message_type}]"

        if digital_employee:
            prefix += f"[employee={digital_employee}]"

        if is_final is not None:
            prefix += f"[final={is_final}]"

        print(
            f"{prefix}\n{message}\n",
            file=sys.stdout,
            flush=True,
        )

    def send_simple(
        self,
        message_type: str,
        message: Any,
        digital_employee: Optional[str] = None,
    ) -> None:
        self.send(
            message_id=None,
            message_type=message_type,
            message=message,
            digital_employee=digital_employee,
            is_final=True,
        )

    def send_partial(
        self,
        message_id: str,
        message_type: str,
        message: Any,
        is_final: Optional[bool] = None,
    ) -> None:
        self.send(
            message_id=message_id,
            message_type=message_type,
            message=message,
            digital_employee=None,
            is_final=is_final,
        )

    def update_agent_type(self, agent_type) -> None:
        self.agent_type = agent_type

    def close(self) -> None:
        pass
