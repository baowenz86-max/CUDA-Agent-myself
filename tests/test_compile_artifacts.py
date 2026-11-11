import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cuda_agent.compile_artifacts import prepare_compile_cache, save_compile_artifacts


class CompileArtifactTests(unittest.TestCase):
    def test_task_isolation_and_archive_preserves_paths_and_binaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = {"TRITON_CACHE_DIR": "/shared/cache", "TRITON_STORE_BINARY_ONLY": "1"}
            cache, env = prepare_compile_cache(root, original)
            other, _ = prepare_compile_cache(root, original)
            self.assertNotEqual(cache, other)
            self.assertEqual(original["TRITON_CACHE_DIR"], "/shared/cache")
            self.assertEqual(env["TRITON_STORE_BINARY_ONLY"], "0")
            for name in ("triton/a/kernel.cubin", "triton/b/kernel.cubin", "triton/a/kernel.ptx",
                         "inductor/helper.so", "triton/a/kernel.ttir", "inductor/output.py"):
                source = cache / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(b"artifact")
            destination = root / "result/torch_compile"
            with patch("cuda_agent.compile_artifacts.shutil.which", return_value="/bin/cuobjdump"), patch(
                "cuda_agent.compile_artifacts.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, "SASS instructions", ""),
            ) as run:
                save_compile_artifacts(cache, destination, env)
            self.assertEqual(run.call_count, 2)
            self.assertEqual((destination / "inductor/helper.so").read_bytes(), b"artifact")
            self.assertEqual((destination / "triton/b/kernel.sass").read_text(), "SASS instructions")
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(manifest["errors"], [])
            self.assertIn("triton/a/kernel.ptx", manifest["files"])

    def test_disassembly_failure_preserves_cubin_and_records_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, env = prepare_compile_cache(root, {})
            (cache / "kernel.cubin").write_bytes(b"binary")
            with patch("cuda_agent.compile_artifacts.shutil.which", return_value="cuobjdump"), patch(
                "cuda_agent.compile_artifacts.subprocess.run",
                side_effect=subprocess.TimeoutExpired("cuobjdump", 60),
            ):
                save_compile_artifacts(cache, root / "saved", env)
            self.assertTrue((root / "saved/kernel.cubin").is_file())
            self.assertFalse((root / "saved/kernel.sass").exists())
            self.assertTrue(json.loads((root / "saved/manifest.json").read_text())["errors"])


if __name__ == "__main__":
    unittest.main()
