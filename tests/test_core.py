import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cuda_agent.api import _apply_files, _extract_json
from cuda_agent.config import OpenAIAPIConfig
from cuda_agent.report import WorkflowReport


class ConfigTests(unittest.TestCase):
    def test_openai_api_config(self) -> None:
        env = {
            "OPENAI_API_KEY": "secret",
            "OPENAI_BASE_URL": "https://api.openai.com/v1/",
            "CUDA_AGENT_MODEL": "gpt-5.6",
            "CUDA_AGENT_API_TIMEOUT": "12",
        }
        with patch.dict(os.environ, env, clear=True):
            config = OpenAIAPIConfig.from_env()
        self.assertEqual(
            config.chat_completions_url,
            "https://api.openai.com/v1/chat/completions",
        )
        self.assertEqual(config.timeout_seconds, 12)
        self.assertEqual(config.model, "gpt-5.6")
        self.assertEqual(config.reasoning_effort, "medium")


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
