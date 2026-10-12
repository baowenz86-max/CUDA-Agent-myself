"""Regression cases for task11–30 failures; most tests need no GPU or torch."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from cuda_agent.api import run_agent, _apply_files, _response_content, _compact, _response_contract
from cuda_agent.candidates import CandidateStore, better
from cuda_agent.config import DeepSeekAPIConfig
from cuda_agent.repair import failure_category
from cuda_agent.validation import check_sources
from cuda_agent.workflow import SEMANTIC_RULES, GENERATION_PROMPTS, generate_cuda, send_feedback
from agent_workdir.utils.operator_contract import validate_contract, contract_call_errors, binding_contract_errors
from agent_workdir.utils.diagnostics import reference_calls
from tests.test_correctness import verification
from scripts.compare_runs import summarize


def contract():
    return {'operators': [{'name': 'op', 'parameters': [
        {'name': 'x', 'kind': 'tensor', 'dtype': 'torch.float32', 'device': 'cuda',
         'shape': '[N,C,L]', 'layout': 'accepts strides'},
        {'name': 'bias', 'kind': 'tensor', 'dtype': 'torch.float32', 'device': 'cuda',
         'shape': '[C]', 'layout': 'contiguous', 'nullable': True, 'optional': True, 'default': None}],
        'returns': {'type': 'tensor', 'dtype': 'torch.float32', 'device': 'cuda', 'shape': 'same as x'},
        'semantics': ['reference omits bias; pass None']}],
        'reference_steps': []}


def source(call='cuda_extension.op(x, None)'):
    return 'import cuda_extension\nclass ModelNew:\n def forward(self,x): return ' + call + '\n'


def response(content, finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {'content': content}}],
            'usage': {'completion_tokens': 12}}


class BiasReference:
    def forward(self, x):
        x = F.conv_transpose1d(x, self.conv.weight, stride=self.conv.stride)
        return x


class FakeResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.value).encode()


class APIRepairTests(unittest.TestCase):
    def run_mock(self, root, values, stage=None, analysis=False):
        config = DeepSeekAPIConfig('test-key', 'https://example.invalid', 'test-model')
        requests = []
        def request(req, **kwargs):
            requests.append(json.loads(req.data))
            value = values[len(requests) - 1]
            if isinstance(value, Exception):
                raise value
            return FakeResponse(value)
        with patch('cuda_agent.api.DeepSeekAPIConfig.from_env', return_value=config), \
             patch('cuda_agent.api.urlopen', side_effect=request):
            result = run_agent('test instructions', str(root), allowed_stage=stage,
                               run_dir=root / 'run', require_analysis=analysis)
        events = [json.loads(x) for x in (root / 'run/api_calls.jsonl').read_text().splitlines()]
        return result, requests, events

    def test_bad_json_corrected_in_same_stage_without_partial_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model_new.py').write_text('original')
            result, requests, events = self.run_mock(root, [response('{bad json'),
                response(json.dumps({'files': {'kernels/custom_ops.cu': '#include <cuda_runtime.h>'}}))], 'cuda')
            self.assertTrue(result)
            self.assertEqual(len(requests), 2)
            self.assertEqual([x['stage'] for x in events], ['cuda', 'cuda'])
            self.assertEqual(events[0]['failure_category'], 'json')
            self.assertEqual(events[1]['correction_attempt'], 1)
            self.assertEqual(events[1]['finish_reason'], 'stop')
            self.assertEqual((root / 'model_new.py').read_text(), 'original')
            self.assertIn('kernels/custom_ops.cu', requests[0]['messages'][0]['content'])
            self.assertNotIn('"model_new.py":', requests[0]['messages'][0]['content'])

    def test_exhausted_scope_corrections_leave_all_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model_new.py').write_text('original')
            bad = response(json.dumps({'files': {'model_new.py': 'wrong owner', 'kernels/x.cpp': 'valid'}}))
            result, requests, events = self.run_mock(root, [bad] * 3, 'binding')
            self.assertFalse(result)
            self.assertEqual(len(requests), 3)
            self.assertEqual((root / 'model_new.py').read_text(), 'original')
            self.assertFalse((root / 'kernels/x.cpp').exists())
            self.assertTrue(all(x['failure_category'] == 'response_validation' for x in events))

    def test_truncation_rejected_even_for_valid_nonempty_json(self):
        with self.assertRaisesRegex(ValueError, 'finish_reason=length'):
            _response_content(response('{"files":{"model_new.py":"truncated"}}', 'length'))

    def test_transport_failure_is_not_a_format_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, requests, events = self.run_mock(Path(temporary), [URLError('offline')])
            self.assertFalse(result)
            self.assertEqual(len(requests), 1)
            self.assertEqual(events[0]['failure_category'], 'transport_or_io')

    def test_nonstring_source_is_corrected_before_repeat_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); run = root / 'run'; run.mkdir()
            (root / 'model_new.py').write_text('original')
            (run / 'repair_context.json').write_text(json.dumps({'different_hypothesis_required': True}))
            analysis = {'hypothesis': 'wrong type', 'evidence': ['failure'], 'changes': ['fix']}
            invalid = response(json.dumps({'analysis': analysis, 'files': {'model_new.py': None}}))
            valid = response(json.dumps({'analysis': analysis, 'files': {'model_new.py': 'changed'}}))
            result, requests, events = self.run_mock(root, [invalid, valid], analysis=True)
            self.assertTrue(result)
            self.assertEqual(len(requests), 2)
            self.assertEqual((root / 'model_new.py').read_text(), 'changed')

    def test_python_requires_valid_metadata_before_applying_any_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = {'files': {'model_new.py': source()}}
            invalid = response(json.dumps(bundle))
            bundle['operator_contract'] = contract()
            result, requests, events = self.run_mock(root, [invalid, response(json.dumps(bundle))], 'python')
            self.assertTrue(result)
            self.assertEqual(len(requests), 2)
            self.assertEqual(json.loads((root / 'run/custom_ops_contract.json').read_text()), contract())

    def test_repair_noop_and_repeated_rejected_source_are_not_applied(self):
        from cuda_agent.api import _reject_repeated_sources
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model_new.py').write_text('current')
            with self.assertRaisesRegex(ValueError, 'no source changes'):
                _reject_repeated_sources({'files': {'model_new.py': 'current'}}, root, None)
            run = root / 'run'; run.mkdir()
            context = {'different_hypothesis_required': True, 'recent_candidates': [
                {'manifest': {'model_new.py': hashlib.sha256(b'rejected').hexdigest()}}]}
            (run / 'repair_context.json').write_text(json.dumps(context))
            with self.assertRaisesRegex(ValueError, 'repeats an evaluated'):
                _reject_repeated_sources({'files': {'model_new.py': 'rejected'}}, root, run)

    def test_metadata_failure_rolls_back_entire_source_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model_new.py').write_text('original')
            run = root / 'run'; run.mkdir()
            destination = run / 'custom_ops_contract.json'
            destination.write_text('old contract')
            original_replace = Path.replace
            def replace(path, target):
                if target == destination:
                    raise OSError('metadata promotion failed')
                return original_replace(path, target)
            with patch.object(Path, 'replace', replace), self.assertRaises(OSError):
                _apply_files({'files': {'model_new.py': 'changed'}}, root,
                             metadata=(destination, contract()))
            self.assertEqual((root / 'model_new.py').read_text(), 'original')
            self.assertEqual(destination.read_text(), 'old contract')


class ContractTests(unittest.TestCase):
    def test_task24_reference_bias_omission_is_explicit(self):
        calls = reference_calls(BiasReference())
        self.assertEqual(calls[0]['bias'], 'None (omitted)')
        self.assertEqual(calls[0]['args'], ['x', 'self.conv.weight'])

    def test_task27_parameter_mismatch_and_unknown_dynamic_calls(self):
        self.assertTrue(contract_call_errors(source('cuda_extension.op(x, None, 2)'), contract()))
        self.assertTrue(contract_call_errors(source('cuda_extension.op(x, 2)'), contract()))
        self.assertTrue(contract_call_errors(source('cuda_extension.op(x, bias=None, wrong=2)'), contract()))
        self.assertFalse(contract_call_errors(source(), contract()))
        self.assertFalse(contract_call_errors(source('cuda_extension.op(*args, **kwargs)'), contract()))

    def test_missing_debug_export_is_checked(self):
        code = source() + ' def debug_forward(self,x): yield "step_001", cuda_extension.missing(x)\n'
        self.assertTrue(any('missing' in x for x in contract_call_errors(code, contract())))

    def test_compiled_pybind_signature_checked_without_invoking_kernel(self):
        def op(*args):
            raise AssertionError('should never execute')
        extension = types.SimpleNamespace(op=op)
        op.__doc__ = 'op(x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor'
        self.assertFalse(binding_contract_errors(extension, contract()))
        op.__doc__ = 'op(x: torch.Tensor, bias: int, batch: int) -> torch.Tensor'
        self.assertTrue(binding_contract_errors(extension, contract()))
        op.__doc__ = None
        self.assertFalse(binding_contract_errors(extension, contract()))

    def test_reference_source_and_checkpoint_order_cannot_be_rewritten(self):
        reference = {'trace_available': True, 'checkpoints': [{'id': 'step_001', 'source': 'x = f(x)'}]}
        with self.assertRaisesRegex(ValueError, 'every reference checkpoint'):
            validate_contract(contract(), reference)
        value = contract()
        value['reference_steps'] = [{'checkpoint': 'step_001', 'source': 'x = f(x)', 'semantics': ['f preserves order']}]
        self.assertEqual(validate_contract(value, reference), value)

    def test_later_generation_cannot_change_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); run = root / 'run'; run.mkdir()
            (root / 'model_new.py').write_text(source())
            (run / 'custom_ops_contract.json').write_text(json.dumps(contract()))
            changed = contract(); changed['operators'][0]['parameters'].pop()
            with self.assertRaisesRegex(ValueError, 'cannot change'):
                _response_contract({'files': {}, 'operator_contract': changed}, root, run, 'binding')

    def test_task29_constructor_allowed_but_torch_computation_still_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'kernels').mkdir()
            (root / 'kernels/custom_ops.cu').write_text('')
            (root / 'model.py').write_text('')
            (root / 'model_new.py').write_text(source('torch.return_types.max((x,x))'))
            self.assertFalse(check_sources(root).errors)
            (root / 'model_new.py').write_text(source('torch.max(x)'))
            self.assertTrue(check_sources(root).errors)


class SelectionTests(unittest.TestCase):
    def candidate(self, kind, matched=0, ratio=.5):
        evidence = verification(ratio)
        evidence['checks'] = evidence['checks'][:1]
        evidence['checks'][0]['error']['kind'] = kind
        evidence['first_failure'] = evidence['checks'][0]
        return {'stage': 'verification', 'verification': evidence,
                'diagnostic': {'status': 'failed', 'matched_checkpoints': matched, 'input_seed': 2027}}

    def test_task14_shape_fix_retained_for_order_repair(self):
        old = self.candidate('tensor_structure')
        fixed_shape = self.candidate('tensor_values', matched=3, ratio=.96)
        self.assertTrue(better(fixed_shape, old))
        self.assertFalse(better(old, fixed_shape))

    def test_interface_fix_is_progress_even_if_forward_then_fails(self):
        old = {'stage': 'interface', 'verification': None}
        new = self.candidate('runtime')
        new['verification']['checks'][0]['stage'] = 'forward'
        self.assertTrue(better(new, old))

    def test_same_error_category_with_improvement_is_not_stagnant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); work = root / 'work'; work.mkdir()
            (work / 'model_new.py').write_text('initial')
            store = CandidateStore(work, root / 'run')
            store.capture('verification', False, 'same error', verification(.5))
            (work / 'model_new.py').write_text('improved')
            store.capture('verification', False, 'same error', verification(.1))
            context = json.loads((root / 'run/repair_context.json').read_text())
            self.assertFalse(context['different_hypothesis_required'])

    def test_task29_corrected_container_can_reveal_child_value_failure(self):
        old = self.candidate('type')
        new = self.candidate('tensor_values', 2)
        new['verification']['checks'][0]['error']['path'] = 'output[0]'
        self.assertTrue(better(new, old))

    def test_lower_final_error_cannot_hide_checkpoint_regression(self):
        old = self.candidate('tensor_values', 3, .5)
        regressed = self.candidate('tensor_values', 0, .1)
        self.assertFalse(better(regressed, old))

    def test_newly_failing_case_cannot_win_by_passing_more_other_cases(self):
        old = {'stage': 'verification', 'verification': verification(.5, 1)}
        new = {'stage': 'verification', 'verification': verification(.1, 3)}
        new['verification']['checks'][0]['status'] = 'failed'
        self.assertFalse(better(new, old))

    def test_different_seed_or_missing_checkpoint_evidence_is_rejected(self):
        old = self.candidate('tensor_values', 1)
        new = self.candidate('tensor_values', 1, .1)
        new['diagnostic']['input_seed'] = 999
        self.assertFalse(better(new, old))

    def test_replaying_previous_input_allows_progress_to_later_failing_seed(self):
        old = self.candidate('tensor_values', 2)
        new = self.candidate('tensor_values', 0, .1)
        new['verification']['checks'][0]['status'] = 'passed'
        later = {'index': 1, 'seed': 2028, 'status': 'failed', 'stage': 'comparison',
                 'error': {'kind': 'tensor_values', 'path': 'output', 'expected': {'shape': [3]}}}
        new['verification']['checks'].append(later)
        new['verification']['first_failure'] = later
        new['diagnostic']['input_seed'] = 2028
        new['diagnostic']['comparison'] = {'status': 'matched', 'input_seed': 2027, 'matched_checkpoints': 4}
        self.assertTrue(better(new, old))
        new['diagnostic'] = {'status': 'unavailable'}
        self.assertFalse(better(new, old))

    def test_best_restore_aligns_reports_contract_and_immutable_failure_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); work = root / 'work'; run = root / 'run'
            work.mkdir(); run.mkdir()
            (work / 'model_new.py').write_text('best')
            (run / 'failure_inputs.pt').write_text('best input')
            (run / 'custom_ops_contract.json').write_text('{"version": "best"}')
            store = CandidateStore(work, run)
            first = self.candidate('tensor_values', 2)
            record = store.capture('verification', False, 'BEST ERROR', first['verification'], first['diagnostic'])
            (work / 'model_new.py').write_text('bad')
            (run / 'replacement.pt').write_text('bad input')
            (run / 'replacement.pt').replace(run / 'failure_inputs.pt')
            (run / 'custom_ops_contract.json').write_text('{"version": "bad"}')
            bad = self.candidate('tensor_values', 0, .9)
            store.capture('verification', False, 'BAD ERROR', bad['verification'], bad['diagnostic'])
            store.restore_best()
            self.assertEqual((work / 'model_new.py').read_text(), 'best')
            self.assertEqual((run / 'failure_inputs.pt').read_text(), 'best input')
            self.assertEqual(Path(record['replay_inputs']).read_text(), 'best input')
            self.assertEqual(json.loads((run / 'diagnostic.json').read_text())['matched_checkpoints'], 2)
            self.assertEqual(json.loads((run / 'custom_ops_contract.json').read_text())['version'], 'best')
            context = json.loads((run / 'repair_context.json').read_text())
            self.assertEqual(context['base_candidate_id'], record['id'])
            self.assertEqual(context['base_manifest'], record['manifest'])


class RoutingTests(unittest.TestCase):
    def test_failures_use_specific_prompts_and_runtime_message_survives_compaction(self):
        for kind, message, expected in [('runtime', 'TypeError: arguments incompatible', 'runtime'),
                                        ('runtime', 'CUDA illegal memory access', 'memory'),
                                        ('tensor_structure', '', 'structure'), ('tensor_values', '', 'numerical')]:
            record = {'stage': 'verification', 'verification': {'first_failure': {'error': {'kind': kind, 'message': message}}}}
            self.assertEqual(failure_category(record), expected)
        compact = _compact({'error': {'kind': 'runtime', 'message': 'bias must be 1D'}, 'traceback': 'noise'})
        self.assertEqual(compact['error']['message'], 'bias must be 1D')
        self.assertNotIn('traceback', compact)
        with patch('cuda_agent.workflow.run_agent', return_value=True) as agent:
            send_feedback('missing export', Path('.'), 'sources')
        self.assertIn('源码检查失败', agent.call_args.args[0])

    def test_prompt_rules_address_logged_semantic_failures(self):
        self.assertIn('atomicAdd', SEMANTIC_RULES)
        self.assertIn('CPU 指针', SEMANTIC_RULES)
        self.assertIn('int64_t', SEMANTIC_RULES)
        self.assertIn('省略 bias', GENERATION_PROMPTS[0][1])
        self.assertIn('cuDNN', GENERATION_PROMPTS[1][1])
        with patch('cuda_agent.workflow.run_agent', return_value=True) as agent:
            self.assertTrue(generate_cuda(Path('.')))
        self.assertEqual([x.kwargs['allowed_stage'] for x in agent.call_args_list], ['python', 'cuda', 'binding'])


class ComparisonTests(unittest.TestCase):
    def test_only_completed_reports_count_and_missing_tasks_are_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / 'task_0011/20261012_100000'; old.mkdir(parents=True)
            (old / 'workflow_report.md').write_text('- Status: **FAILED**\n- Duration: `1.0s`')
            latest = root / 'task_0011/20261012_110000'; latest.mkdir()
            (latest / 'workflow_report.md').write_text('- Status: **SUCCESS**\n- Duration: `2.0s`')
            (root / 'task_0011/20261012_120000').mkdir()
            summary = summarize(root, 11, 12)
            self.assertEqual(summary['successful_tasks'], 1)
            self.assertEqual(summary['success_rate'], 1)
            self.assertEqual(summary['missing_tasks'], [12])
            self.assertEqual(summary['summed_task_seconds'], 2)
            self.assertIsNone(summary['effective_repair_rate'])


if __name__ == '__main__':
    unittest.main()
