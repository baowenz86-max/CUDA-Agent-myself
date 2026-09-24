import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cuda_agent.api import _apply_files, _extract_json
from cuda_agent.config import SchoolAPIConfig
from cuda_agent.report import WorkflowReport


class ConfigTests(unittest.TestCase):
    def test_school_api_config(self) -> None:
        env = {
            "CUDA_AGENT_API_KEY": "secret",
            "CUDA_AGENT_BASE_URL": "https://school.example/v1/",
            "CUDA_AGENT_MODEL": "school-model",
            "CUDA_AGENT_API_TIMEOUT": "12",
        }
        with patch.dict(os.environ, env, clear=True):
            config = SchoolAPIConfig.from_env()
        self.assertEqual(
            config.chat_completions_url,
            "https://school.example/v1/chat/completions",
        )
        self.assertEqual(config.timeout_seconds, 12)


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
            report.add_stage("generation", False, 0.5, "school API")
            report_path = report.write(False, "request failed")
            content = report_path.read_text(encoding="utf-8")
            self.assertIn("FAILED", content)
            self.assertIn("request failed", content)


if __name__ == "__main__":
    unittest.main()
