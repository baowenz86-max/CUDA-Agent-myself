"""Configure logging for CUDA Agent entry points."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def configure_logging(log_root: Path, task_id: int, verbose: bool = False) -> Path:
    """Configure console logging and a detailed per-run log file."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = log_root / f"task_{task_id:04d}" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "workflow.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    return log_path
