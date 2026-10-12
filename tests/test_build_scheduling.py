import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from cuda_agent.api import _apply_files
from cuda_agent.validation import check_sources
from cuda_agent.workflow import COMPILE_COMMAND, GPUPool, run_command, save_result


class SingleSourceTests(unittest.TestCase):
    def test_multiple_sources_rejected_without_partial_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                _apply_files({'files': {'kernels/custom_ops.cu': 'valid', 'kernels/extra.cu': 'invalid'}}, root, 'cuda')
            self.assertFalse((root / 'kernels/custom_ops.cu').exists())

    def test_zero_multiple_and_nested_sources_fail_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model_new.py').write_text('class ModelNew:\n def forward(self,x): return x')
            (root / 'kernels').mkdir()
            self.assertTrue(check_sources(root).errors)
            (root / 'kernels/custom_ops.cu').write_text('')
            self.assertFalse(check_sources(root).errors)
            (root / 'kernels/nested').mkdir()
            (root / 'kernels/nested/extra.cu').write_text('')
            self.assertTrue(check_sources(root).errors)
            with self.assertRaises(ValueError):
                save_result(0, root, {})


class SchedulingTests(unittest.TestCase):
    def test_compile_does_not_acquire_busy_gpu_and_logs_separate_timings(self):
        pool = GPUPool(['3'])
        with pool.acquire():
            with patch('cuda_agent.workflow.subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'ok')) as run:
                with self.assertLogs('cuda_agent.workflow', level='INFO') as logs:
                    self.assertTrue(run_command(COMPILE_COMMAND, Path('.'), {}, pool)[0])
            self.assertNotIn('CUDA_VISIBLE_DEVICES', run.call_args.kwargs['env'])
        self.assertTrue(any('resource=compile' in line and 'wait=' in line and 'execution=' in line for line in logs.output))

    def test_gpu_execution_receives_lease_and_returns_it_on_error(self):
        pool = GPUPool(['3'])
        with patch('cuda_agent.workflow.subprocess.run', side_effect=OSError('failed')) as run:
            with self.assertRaises(OSError):
                run_command(('python', '-m', 'utils.verification'), Path('.'), {}, pool)
        self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '3')
        self.assertEqual(pool._available, ['3'])


class IncrementalBuildTests(unittest.TestCase):
    def test_python_reuses_binary_native_changes_rebuild_and_failure_removes_binary(self):
        fake = types.ModuleType('torch.utils.cpp_extension')
        fake.torch = types.SimpleNamespace(__version__='test')
        fake._get_cuda_arch_flags = lambda: ['-gencode=arch=compute_80,code=sm_80']
        builds = []
        def load(**kwargs):
            builds.append(kwargs)
            (Path(kwargs['build_directory']) / 'cuda_extension.so').write_bytes(b'binary')
            (Path(kwargs['build_directory']) / 'cached.o').write_bytes(b'object')
        fake.load = load
        modules = {'torch': types.ModuleType('torch'), 'torch.utils': types.ModuleType('torch.utils'),
                   'torch.utils.cpp_extension': fake}
        modules['torch'].utils = modules['torch.utils']
        modules['torch.utils'].cpp_extension = fake
        source = Path(__file__).resolve().parents[1] / 'agent_workdir/utils/compile.py'
        spec = importlib.util.spec_from_file_location('compile_under_test', source)
        compiler = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', modules), patch.dict(os.environ):
            spec.loader.exec_module(compiler)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'kernels').mkdir()
                (root / 'kernels/custom_ops.cu').write_text('kernel')
                (root / 'binding.cpp').write_text('binding')
                (root / 'binding_registry.h').write_text('header')
                (root / 'gcc').write_text('compiler')
                (root / 'bin').mkdir()
                (root / 'bin/nvcc').write_text('nvcc')
                compiler.HOST_CXX = str(root / 'gcc')
                compiler.CUDA_HOME = str(root)
                previous = Path.cwd()
                try:
                    os.chdir(root)
                    self.assertEqual(compiler.compile_kernels(), 0)
                    (root / 'model_new.py').write_text('changed Python only')
                    self.assertEqual(compiler.compile_kernels(), 0)
                    self.assertEqual(len(builds), 1)
                    for name in ('kernels/custom_ops.cu', 'binding.cpp', 'binding_registry.h'):
                        path = root / name
                        path.write_text(path.read_text() + ' changed')
                        self.assertEqual(compiler.compile_kernels(), 0)
                    self.assertEqual(len(builds), 4)
                    with patch.dict(os.environ, {'CUDA_AGENT_FAST_MATH': '1'}):
                        self.assertEqual(compiler.compile_kernels(), 0)
                    self.assertEqual(len(builds), 5)
                    # Returning to prior flags must rebuild, and corrupt binaries cannot hit the cache.
                    self.assertEqual(compiler.compile_kernels(), 0)
                    (root / 'build/forced_compile/cuda_extension.so').write_bytes(b'corrupt')
                    self.assertEqual(compiler.compile_kernels(), 0)
                    self.assertEqual(len(builds), 7)
                    self.assertTrue((root / 'build/forced_compile/cached.o').exists())
                    (root / 'kernels/custom_ops.cu').write_text('broken')
                    with patch.object(fake, 'load', side_effect=RuntimeError('compile error')):
                        self.assertEqual(compiler.compile_kernels(), 1)
                    self.assertFalse((root / 'cuda_extension.so').exists())
                    self.assertFalse((root / 'build/forced_compile/cache_manifest.json').exists())
                finally:
                    os.chdir(previous)
