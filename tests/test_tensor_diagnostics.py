"""Runtime regression tests; CPU tests require torch, CUDA tests are opt-in."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from agent_workdir.utils.diagnostics import ComparisonFailure, check_equal, tensor_error


if torch is not None:
    from agent_workdir.utils.verification import diagnose, main as verify_main, seed_all

    class Reference(torch.nn.Module):
        def forward(self, x):
            x = x * 2
            x = x + 1
            return x

    class Generated(Reference):
        def __init__(self, bad=False, forbidden=False):
            super().__init__()
            self.bad = bad
            self.forbidden = forbidden

        def debug_forward(self, x):
            x = torch.nn.functional.relu(x) if self.forbidden else x * (3 if self.bad else 2)
            yield 'step_001', x
            x = x + 1
            yield 'step_002', x
            yield 'step_003', x


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class TensorDiagnosticsTests(unittest.TestCase):
    def test_int64_promotion_and_value_error(self):
        with self.assertRaises(ComparisonFailure) as raised:
            check_equal(torch.tensor([1.], dtype=torch.float32), torch.tensor([1], dtype=torch.int64))
        self.assertEqual(raised.exception.detail['kind'], 'tensor_structure')
        detail = tensor_error(torch.tensor([0, 0, 0]), torch.tensor([0, 1, 1]))
        self.assertEqual(detail['mismatched'], 2)
        self.assertAlmostEqual(detail['mismatch_ratio'], 2 / 3)
        self.assertEqual(detail['samples'][0]['index'], [1])

    def test_nan_inf_keep_original_rejection_and_json_safe_metrics(self):
        actual = torch.tensor([float('nan'), float('inf'), 0.])
        expected = torch.tensor([float('nan'), float('inf'), 1.])
        with self.assertRaises(ComparisonFailure):
            check_equal(actual, expected)
        detail = tensor_error(actual, expected)
        self.assertEqual(detail['actual_nan'], 1)
        self.assertEqual(detail['expected_inf'], 1)
        self.assertEqual(detail['mismatched'], 2)
        self.assertEqual(detail['max_absolute_error'], 'inf')

    def test_first_mismatched_checkpoint_and_forbidden_debug_call(self):
        reference, actual = Reference(), Generated(bad=True)
        fixture = {'state_dict': {}}
        failure = diagnose(reference, actual, fixture, [torch.tensor([1.])], 2027, 'cpu')
        self.assertEqual(failure['checkpoint']['id'], 'step_001')
        self.assertEqual(failure['matched_checkpoints'], 0)
        matched = diagnose(reference, Generated(), fixture, [torch.tensor([1.])], 2027, 'cpu')
        self.assertEqual(matched['status'], 'matched')
        forbidden = diagnose(reference, Generated(forbidden=True), fixture, [torch.tensor([1.])], 2027, 'cpu')
        self.assertEqual(forbidden['status'], 'unavailable')
        self.assertIn('not allowed', forbidden['reason'])

    def test_task_1_relu_position_is_inplace_not_threshold(self):
        x = torch.tensor([0., .01, .1, 1.])
        expected = torch.nn.functional.relu(x.clone(), .2)
        wrong = x.clamp_min(.2)
        with self.assertRaises(ComparisonFailure):
            check_equal(wrong, expected)
        check_equal(x.clamp_min(0), expected)

    def test_task_2_group_norm_axis_after_transpose(self):
        seed_all(2026)
        x = torch.randn(2, 4, 8, 3).ceil().transpose(1, 2).contiguous()
        expected = torch.nn.functional.group_norm(x, 2)
        wrong = torch.nn.functional.group_norm(x.transpose(1, 2).contiguous(), 2).transpose(1, 2).contiguous()
        with self.assertRaises(ComparisonFailure):
            check_equal(wrong, expected)
        check_equal(torch.nn.functional.group_norm(x, 2), expected)

    def test_task_4_tan_is_not_tanh(self):
        x = torch.tensor([.5, 1., 2.])
        with self.assertRaises(ComparisonFailure):
            check_equal(torch.tanh(x * .5), torch.tan(x * .5))

    def test_task_5_conv_transpose_weight_layout(self):
        x = torch.tensor([[[1., 2.], [3., 4.]]])
        weight = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        expected = torch.nn.functional.conv_transpose1d(x, weight, stride=2, padding=1, output_padding=1)
        def compute(extra_channel_offset):
            result = torch.zeros_like(expected)
            flat = torch.cat([weight.flatten(), torch.zeros(2)])
            for cout in range(2):
                for out in range(expected.shape[-1]):
                    for k in range(3):
                        source = out + 1 - k
                        if source % 2 == 0 and 0 <= source // 2 < 2:
                            for cin in range(2):
                                index = cin * 2 * 3 + cout * 3 + k + (cin if extra_channel_offset else 0)
                                result[0, cout, out] += x[0, cin, source // 2] * flat[index]
            return result
        with self.assertRaises(ComparisonFailure):
            check_equal(compute(True), expected)
        check_equal(compute(False), expected)

    def test_task_9_histogram_excludes_values_outside_range(self):
        x = torch.tensor([-3., -1., .2, 2., 5., 11.])
        expected = torch.histc(x, bins=20, min=0, max=10)
        with self.assertRaises(ComparisonFailure):
            check_equal(torch.histc(x.clamp(0, 10), bins=20, min=0, max=10), expected)
        check_equal(torch.histc(x[(x >= 0) & (x <= 10)], bins=20, min=0, max=10), expected)

    def test_debug_match_cannot_override_normal_forward_failure_and_replay(self):
        # A file-backed fixture also exercises actual import, state loading and replay.
        import json
        import sys
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model.py').write_text('''import torch
class Model(torch.nn.Module):
    def forward(self,x):
        x = x * 2
        return x
def get_inputs(): return [torch.randn(3)]
def get_init_inputs(): return []
''')
            (root / 'model_new.py').write_text('''from model import Model
class ModelNew(Model):
    def forward(self,x): return x * 3
    def debug_forward(self,x):
        x = x * 2
        yield "step_001", x
        yield "step_002", x
''')
            modules = {}
            with patch.dict(os.environ, {'CUDA_AGENT_RUN_DIR': str(root)}), patch.dict(sys.modules, clear=False):
                for name in ('model', 'model_new'):
                    spec = importlib.util.spec_from_file_location(name, root / (name + '.py'))
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    spec.loader.exec_module(module)
                    modules[name] = module
                self.assertEqual(verify_main(['--device', 'cpu']), 1)
                original = json.loads((root / 'verification.json').read_text())
                self.assertEqual(len(original['checks']), 5)
                self.assertEqual(verify_main(['--device', 'cpu', '--diagnose']), 0)
                self.assertEqual(json.loads((root / 'diagnostic.json').read_text())['status'], 'matched')
                self.assertEqual(json.loads((root / 'verification.json').read_text())['status'], 'failed')
                self.assertEqual(verify_main(['--device', 'cpu', '--replay', str(root / 'failure_inputs.pt')]), 1)
                replay = json.loads((root / 'replay_verification.json').read_text())
                self.assertEqual(original['first_failure']['error'], replay['first_failure']['error'])


@unittest.skipUnless(torch is not None and os.environ.get('CUDA_AGENT_GPU_TESTS') == '1', 'Set CUDA_AGENT_GPU_TESTS=1 on a CUDA host')
class CUDACoverageTests(unittest.TestCase):
    def test_task_6_grid_cap_requires_stride_and_all_outputs_written(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA GPU unavailable')
        from torch.utils.cpp_extension import load_inline
        cuda = r'''
#include <torch/types.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
__global__ void coverage(int64_t* out, int64_t count, bool stride) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (stride) {
        for (; i < count; i += (int64_t)blockDim.x * gridDim.x) out[i] = 1;
    } else if (i < count) out[i] = 1;
}
torch::Tensor launch(int64_t count, bool stride) {
    auto out = torch::full({count}, -1, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    coverage<<<65535,256,0,c10::cuda::getCurrentCUDAStream().stream()>>>(out.data_ptr<int64_t>(),count,stride);
    return out;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            extension = load_inline(name='coverage_regression', cpp_sources='torch::Tensor launch(int64_t count, bool stride);',
                                    cuda_sources=cuda, functions=['launch'], build_directory=directory, with_cuda=True)
            count = 65535 * 256 + 1024
            bad = extension.launch(count, False)
            self.assertTrue((bad[-1024:] == -1).all().item())
            good = extension.launch(count, True)
            self.assertTrue((good == 1).all().item())


if __name__ == '__main__':
    unittest.main()
