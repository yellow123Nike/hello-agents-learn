from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Planner:
    """
    计划类：用于管理和更新任务计划
    """
    # --------- fields ----------
    # 计划标题
    title: Optional[str] = None
    # 计划步骤列表
    steps: List[str] = None
    # 步骤状态列表
    step_status: List[str] = None
    # 步骤备注列表
    notes: List[str] = None
    # 计划步骤列表
    steps_res: List[str] = None

    # 创建新计划
    @staticmethod
    def create(title: str, steps: List[str]):
        status = []
        notes = []
        steps_res = []
        for _ in steps:
            status.append("not_started")
            notes.append("")
            steps_res.append("")
        return Planner(
            title=title,
            steps=steps,
            step_status=status,
            notes=notes,
            steps_res=steps_res,
        )

    # 更新计划：标题+步骤
    def update(self, title: Optional[str], new_steps: Optional[List[str]]):
        if title is not None:
            self.title = title

        if new_steps is not None:
            new_statuses = []
            new_notes = []

            for i, step in enumerate(new_steps):
                if (
                    self.steps is not None
                    and i < len(self.steps)
                    and step == self.steps[i]
                ):
                    new_statuses.append(self.step_status[i])
                    new_notes.append(self.notes[i])
                else:
                    new_statuses.append("not_started")
                    new_notes.append("")

            self.steps = new_steps
            self.step_status = new_statuses
            self.notes = new_notes

    # 更新指定步骤的状态与备注
    def update_step_status(
        self,
        step_index: int,
        status: Optional[str],
        note: Optional[str],
    ):
        if step_index < 0 or step_index >= len(self.steps):
            raise ValueError(f"Invalid step index: {step_index}")

        if status is not None:
            self.step_status[step_index] = status

        if note is not None:
            self.notes[step_index] = note

    # 更新指定步骤的结果
    def update_step_res(
        self,
        res: Optional[str],
    ):
        for i in range(len(self.steps)):
            if self.step_status[i] == "in_progress":
                if res is not None:
                    self.steps_res[i] = res

    #  获取当前正在执行的步骤
    def get_current_step(self) -> str:
        for i in range(len(self.steps)):
            if self.step_status[i] == "in_progress":
                return self.steps[i]
        return ""

    # 将当前步骤标记为 completed，下一个步骤标记为 in_progress
    def step_plan(self):
        if not self.steps:
            return

        if not self.get_current_step():
            self.update_step_status(0, "in_progress", "")
            return

        for i in range(len(self.steps)):
            if self.step_status[i] == "in_progress":
                self.update_step_status(i, "completed", "")
                if i + 1 < len(self.steps):
                    self.update_step_status(i + 1, "in_progress", "")
                break

    # 格式化计划显示

    def format(self) -> str:
        lines = []
        lines.append(f"Plan: {self.title}")
        lines.append("Steps:")

        for i, step in enumerate(self.steps):
            status = self.step_status[i]
            note = self.notes[i]

            lines.append(f"{i + 1}. [{status}] {step}")
            if note:
                lines.append(f"   Notes: {note}")

        return "\n".join(lines)
