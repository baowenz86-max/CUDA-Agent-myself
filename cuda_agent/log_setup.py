"""Configure logging for CUDA Agent entry points."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path


CURRENT_TASK_ID: ContextVar[int | None] = ContextVar("current_task_id", default=None)


class _TaskFilter(logging.Filter):
    def __init__(self, task_id: int) -> None:
        super().__init__()
        self.task_id = task_id

    def filter(self, record: logging.LogRecord) -> bool:
        return CURRENT_TASK_ID.get() == self.task_id


class _TaskFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.task_id = CURRENT_TASK_ID.get()
        return super().format(record)


def configure_logging(log_root: Path, task_id: int, verbose: bool = False) -> Path:
    """Configure console logging and a detailed per-run log file."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = log_root / f"task_{task_id:04d}" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "workflow.log"

    formatter = _TaskFormatter(
        "%(asctime)s | task=%(task_id)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_TaskFilter(task_id))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    if not any(getattr(handler, "_cuda_agent_console", False)
               for handler in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        console_handler.setFormatter(formatter)
        console_handler._cuda_agent_console = True
        root_logger.addHandler(console_handler)
    return log_path
