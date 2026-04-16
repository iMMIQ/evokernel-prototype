from __future__ import annotations

from dataclasses import dataclass, field

from evokernel.orchestrator.run_report import RunReport


@dataclass(slots=True)
class CurriculumReport:
    task_reports: list[tuple[str, RunReport]] = field(default_factory=list)
    strategy: str = "scratch"

    @property
    def task_ids(self) -> list[str]:
        return [task_id for task_id, _ in self.task_reports]

    @property
    def solved_count(self) -> int:
        return sum(
            1 for _, report in self.task_reports if report.best_candidate is not None
        )

    @property
    def total_count(self) -> int:
        return len(self.task_reports)
