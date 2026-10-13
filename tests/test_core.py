import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from cuda_agent.api import _apply_files, _extract_json
from cuda_agent.config import DeepSeekAPIConfig
from cuda_agent.report import WorkflowReport
from cuda_agent.workflow import task_ids_from_args


class ConfigTests(unittest.TestCase):
    def test_deepseek_api_config(self) -> None:
        env = {
            "DEEPSEEK_API_KEY": "secret",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com/",
            "CUDA_AGENT_MODEL": "deepseek-v4-pro",
            "CUDA_AGENT_API_TIMEOUT": "12",
        }
        with patch.dict(os.environ, env, clear=True):
            config = DeepSeekAPIConfig.from_env()
        self.assertEqual(
            config.chat_completions_url,
            "https://api.deepseek.com/chat/completions",
        )
        self.assertEqual(config.timeout_seconds, 12)
        self.assertEqual(config.model, "deepseek-v4-pro")
        self.assertEqual(config.thinking_mode, "enabled")
        self.assertEqual(config.reasoning_effort, "low")


class APIResponseTests(unittest.TestCase):
    def test_response_is_validated_before_any_file_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            (workdir / "model_new.py").write_text("original", encoding="utf-8")
            response = {
                "files": {"model_new.py": "changed", "../escape.cu": "invalid"}
            }
            with self.assertRaises(ValueError):
                _apply_files(response, workdir)
            self.assertEqual(
                (workdir / "model_new.py").read_text(encoding="utf-8"), "original"
            )

    def test_markdown_wrapped_json_is_accepted(self) -> None:
        data = _extract_json(
            "```json\n" + json.dumps({"files": {"model_new.py": "ok"}}) + "\n```"
        )
        self.assertEqual(data["files"]["model_new.py"], "ok")


class ReportTests(unittest.TestCase):
    def test_failure_report_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "workflow.log"
            log_path.touch()
            report = WorkflowReport(3, log_path)
            report.add_stage("generation", False, 0.5, "DeepSeek API")
            report_path = report.write(False, "request failed")
            content = report_path.read_text(encoding="utf-8")
            self.assertIn("FAILED", content)
            self.assertIn("request failed", content)


class TaskSelectionTests(unittest.TestCase):
    def test_batch_uses_shared_repair_config(self):
        from cuda_agent.workflow import WorkflowConfig, main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / 'template'
            work.mkdir()
            config = WorkflowConfig(workdir=work, repair_attempts=3)
            with patch('cuda_agent.workflow.REPO_ROOT', root), \
                 patch('cuda_agent.workflow.WorkflowConfig', return_value=config) as factory, \
                 patch('cuda_agent.workflow.parse_args', return_value=Namespace(task_start=2, task_end=3)), \
                 patch('cuda_agent.workflow.run_task', return_value=True) as task, \
                 patch.dict(os.environ, {'CUDA_AGENT_GENERATION_WORKERS': '1'}):
                self.assertEqual(main(), 0)
            self.assertEqual(task.call_count, 2)
            self.assertEqual(factory.call_args_list[1].kwargs,
                             {'workdir': root / '.task_workdirs/task_0002', 'repair_attempts': 3})
            self.assertEqual(factory.call_args_list[2].kwargs,
                             {'workdir': root / '.task_workdirs/task_0003', 'repair_attempts': 3})

    def test_range_is_inclusive(self) -> None:
        args = Namespace(task_id=None, task_start=0, task_end=10)
        self.assertEqual(list(task_ids_from_args(args)), list(range(11)))

    def test_default_task_is_zero(self) -> None:
        args = Namespace(task_id=None, task_start=None, task_end=None)
        self.assertEqual(list(task_ids_from_args(args)), [0])

    def test_task_id_and_range_are_mutually_exclusive(self) -> None:
        args = Namespace(task_id=0, task_start=0, task_end=10)
        with self.assertRaises(ValueError):
            task_ids_from_args(args)


if __name__ == "__main__":
    unittest.main()
