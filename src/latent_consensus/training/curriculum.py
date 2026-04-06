"""课程训练调度最小骨架。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    start_epoch: int
    label: str


class CurriculumSchedule:
    def __init__(self, stages: list[CurriculumStage]) -> None:
        if not stages:
            raise ValueError("至少需要一个阶段")
        self.stages = sorted(stages, key=lambda stage: stage.start_epoch)

    def stage_for_epoch(self, epoch: int) -> CurriculumStage:
        current_stage = self.stages[0]
        for stage in self.stages:
            if epoch >= stage.start_epoch:
                current_stage = stage
        return current_stage
