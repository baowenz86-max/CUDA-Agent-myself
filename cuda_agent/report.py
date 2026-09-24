"""Collect workflow events and render run reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    duration_seconds: float
    details: str = ""


@dataclass
class WorkflowReport:
    task_id: int
    log_path: Path
    started_at: datetime = field(default_factory=datetime.now)
    stages: list[StageResult] = field(default_factory=list)
    operators: str = "unknown"

    def add_stage(
        self, name: str, success: bool, duration_seconds: float, details: str = ""
    ) -> None:
        self.stages.append(
            StageResult(name, "PASS" if success else "FAIL", duration_seconds, details)
        )

    def write(self, success: bool, error: str = "") -> Path:
        finished_at = datetime.now()
        duration = (finished_at - self.started_at).total_seconds()
        rows = [
            f"| {stage.name} | {stage.status} | {stage.duration_seconds:.2f}s | "
            f"{stage.details.replace('|', '\\|')} |"
            for stage in self.stages
        ]
        if not rows:
            rows.append("| — | — | — | No stage completed |")

        content = f"""# CUDA Agent Workflow Report

- Task ID: `{self.task_id}`
- Status: **{'SUCCESS' if success else 'FAILED'}**
- Operators: `{self.operators}`
- Started: `{self.started_at.isoformat(timespec='seconds')}`
- Finished: `{finished_at.isoformat(timespec='seconds')}`
- Duration: `{duration:.2f}s`
- Full log: `{self.log_path.name}`

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
{chr(10).join(rows)}

## Failure Summary

{error if error else 'None.'}

## Reproduction

```bash
python main.py --task-id {self.task_id}
```
"""
        report_path = self.log_path.parent / "workflow_report.md"
        report_path.write_text(content, encoding="utf-8")
        return report_path
