import ast
import json
import random
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from agent_workdir.utils.diagnostics import trace_function, seed_all, transform_tree
from cuda_agent.api import _apply_files, _build_context, _validate_analysis
from cuda_agent.candidates import CandidateStore, better
from cuda_agent.report import WorkflowReport
from cuda_agent.validation import check_sources
from cuda_agent.workflow import WorkflowConfig, execute_workflow


class TraceTests(unittest.TestCase):
    def test_preserves_tuple_assignment_inplace_and_final_return(self):
        source = '''def forward(self, x):
    "docstring"
    a, b = x, x + 1
    a = a * 2
    return (a, b)
'''
        tree, stages = trace_function(source)
        namespace = {}
        exec(compile(tree, '<test>', 'exec'), namespace)
        trace = list(namespace['_reference_debug_forward'](None, 3))
        self.assertEqual(trace, [('step_001', (3, 4)), ('step_002', 6), ('step_003', (6, 4))])
        self.assertEqual(stages[-1]['source'], 'return (a, b)')

    def test_rejects_control_flow_and_mutating_attribute_targets(self):
        for source in ('def forward(self,x):\n if x: return x\n return 0',
                       'def forward(self,x):\n self.x = x\n return x'):
            with self.assertRaises(ValueError):
                trace_function(source)

    def test_reproducible_seed_and_tuple_structure(self):
        class Tensor:
            pass
        calls = []
        fake_torch = types.SimpleNamespace(Tensor=Tensor, manual_seed=calls.append,
                                          cuda=types.SimpleNamespace(is_available=lambda: False))
        with patch.dict('sys.modules', {'torch': fake_torch}):
            seed_all(17)
            before = random.random()
            seed_all(17)
            self.assertEqual(before, random.random())
            self.assertEqual(calls, [17, 17])
            value = ({'a': Tensor()}, [Tensor()])
            changed = transform_tree(value, lambda t: 'tensor')
            self.assertEqual(changed, ({'a': 'tensor'}, ['tensor']))
            self.assertIsInstance(changed, tuple)


class InterfaceTests(unittest.TestCase):
    def test_generation_owner_enforced_before_any_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model_new.py').write_text('original')
            with self.assertRaises(ValueError):
                _apply_files({'files': {'model_new.py': 'modified', 'kernels/custom_ops.cu': 'bad'}}, root, 'python')
            self.assertEqual((root / 'model_new.py').read_text(), 'original')

    def test_repair_analysis_is_required_and_validated(self):
        with self.assertRaises(ValueError):
            _validate_analysis({'files': {}}, True)
        with self.assertRaises(ValueError):
            _validate_analysis({'analysis': {'hypothesis': 'guess', 'evidence': [], 'changes': ['x']}}, True)
        analysis = {'hypothesis': 'wrong index', 'evidence': ['37.8% mismatch'], 'changes': ['grid-stride loop']}
        self.assertEqual(_validate_analysis({'analysis': analysis}, True), analysis)

    def test_bundle_is_rolled_back_on_promotion_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model_new.py').write_text('original')
            real_replace = Path.replace
            count = 0
            def replace(path, target):
                nonlocal count
                count += 1
                if count == 2:
                    raise OSError('simulated promotion failure')
                return real_replace(path, target)
            with patch.object(Path, 'replace', replace), self.assertRaises(OSError):
                _apply_files({'files': {'model_new.py': 'modified', 'kernels/custom_ops.cu': 'new'}}, root)
            self.assertEqual((root / 'model_new.py').read_text(), 'original')
            self.assertFalse((root / 'kernels/custom_ops.cu').exists())

    def test_missing_binding_and_obvious_prohibited_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model.py').write_text('')
            (root / 'kernels').mkdir()
            (root / 'kernels/custom_ops.cu').write_text('')
            (root / 'model_new.py').write_text('''import cuda_extension
class ModelNew:
    def forward(self,x):
        return cuda_extension.missing(F.relu(x))
''')
            errors = check_sources(root).errors
            self.assertTrue(any('no visible pybind' in e for e in errors))
            self.assertTrue(any('functional' in e for e in errors))

    def test_grid_coverage_warning_is_not_a_proof_or_a_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model.py').write_text('torch.tan(x)')
            (root / 'model_new.py').write_text('class ModelNew:\n def forward(self,x): return x')
            (root / 'kernels').mkdir()
            (root / 'kernels/custom_ops.cu').write_text('if (grid_size > 65535) grid_size = 65535; float y = tanhf(x);')
            result = check_sources(root)
            self.assertFalse(result.errors)
            self.assertEqual(len(result.warnings), 2)

    def test_readonly_runtime_contract_is_in_api_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'run'
            run.mkdir()
            (run / 'task_contract.json').write_text('{"checkpoint":"step_001"}')
            (root / 'binding_registry.h').write_text('registry truth')
            context = _build_context(root, run)
            self.assertIn('registry truth', context)
            self.assertIn('step_001', context)


def verification(ratio, passed=0):
    return {'seed': 2026, 'status': 'failed', 'checks': [
        {'index': i, 'seed': 2027 + i, 'status': 'passed' if i < passed else 'failed',
         'stage': 'comparison', 'error': {'kind': 'tensor_values', 'path': 'output',
             'expected': {'shape': [3]}, 'mismatch_ratio': ratio, 'max_absolute_error': ratio}}
        for i in range(5)]}


class CandidateTests(unittest.TestCase):
    def test_common_cases_must_improve_without_new_failures(self):
        old = {'stage': 'verification', 'verification': verification(.5)}
        improved = {'stage': 'verification', 'verification': verification(.2)}
        self.assertTrue(better(improved, old))
        self.assertFalse(better(old, improved))
        self.assertFalse(better(old, old))
        incomplete = {'stage': 'verification', 'verification': verification(.1)}
        incomplete['verification']['checks'].pop()
        self.assertFalse(better(incomplete, old))

    def test_restore_removes_new_sources_and_preserves_native_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / 'work'
            work.mkdir()
            (work / 'kernels').mkdir()
            (work / 'model_new.py').write_text('best')
            (work / 'kernels/best.cu').write_text('best kernel')
            store = CandidateStore(work, root / 'run')
            store.capture('verification', False, 'mismatch', verification(.2))
            native_mtime = (work / 'kernels/best.cu').stat().st_mtime_ns
            (work / 'model_new.py').write_text('regressed')
            (work / 'kernels/new.cu').write_text('new kernel')
            (work / 'cuda_extension.so').write_text('stale')
            (work / 'build').mkdir()
            (work / 'build/stale.o').write_text('stale')
            rejected = store.capture('verification', False, 'mismatch', verification(.5))
            self.assertEqual(rejected['decision'], 'rejected')
            store.restore_best()
            self.assertEqual((work / 'model_new.py').read_text(), 'best')
            self.assertFalse((work / 'kernels/new.cu').exists())
            self.assertEqual((work / 'kernels/best.cu').stat().st_mtime_ns, native_mtime)
            self.assertTrue((work / 'cuda_extension.so').exists())
            self.assertTrue((work / 'build/stale.o').exists())
            self.assertEqual(len((root / 'run/history.jsonl').read_text().splitlines()), 2)

    def test_snapshot_replay_inputs_survive_atomic_fixture_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work, run = root / 'work', root / 'run'
            work.mkdir(); run.mkdir()
            (work / 'model_new.py').write_text('source')
            (run / 'failure_inputs.pt').write_text('old input')
            store = CandidateStore(work, run)
            evidence = verification(.5)
            evidence['first_failure'] = evidence['checks'][0]
            record = store.capture('verification', False, 'fail', evidence)
            (run / 'replacement.pt').write_text('new input')
            (run / 'replacement.pt').replace(run / 'failure_inputs.pt')
            self.assertEqual(Path(record['replay_inputs']).read_text(), 'old input')


class WorkflowTests(unittest.TestCase):
    def test_repair_recompiles_and_zero_exit_without_five_checks_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / 'work'
            work.mkdir()
            report = WorkflowReport(0, root / 'workflow.log')
            run = root / 'correctness'
            commands, bases = [], []
            calls = 0
            def command(argv, workdir, env, pool):
                nonlocal calls
                commands.append(tuple(argv))
                if argv[-1] == 'utils.verification':
                    calls += 1
                    data = verification(.5 if calls == 1 else .8)
                    # A forged successful process with incomplete results cannot pass.
                    data['status'] = 'passed'
                    (run / 'verification.json').write_text(json.dumps(data))
                return True, 'command output'
            def generate(*args):
                (work / 'model_new.py').write_text('best original')
                return True
            def repair(*args):
                bases.append((work / 'model_new.py').read_text())
                (work / 'model_new.py').write_text('bad repair')
                return True
            with patch('cuda_agent.workflow.load_task', return_value={'code': '', 'ops': []}), \
                 patch('cuda_agent.workflow.generate_cuda', side_effect=generate), \
                 patch('cuda_agent.workflow.send_feedback', side_effect=repair), \
                 patch('cuda_agent.workflow.check_sources', return_value=types.SimpleNamespace(errors=[], warnings=[], as_dict=lambda: {})), \
                 patch('cuda_agent.workflow.run_command', side_effect=command):
                success = execute_workflow(0, WorkflowConfig(work, 1, 3), report, None)
            self.assertFalse(success)
            self.assertEqual(bases, ['best original', 'best original'])
            self.assertEqual((work / 'model_new.py').read_text(), 'best original')
            self.assertEqual(sum(c == ('bash', 'utils/compile.sh') for c in commands), 3)
            self.assertTrue(report.failure_reason)
            self.assertFalse(any(c[-1] == 'utils.profiling' for c in commands))

class AdditionalWorkflowTests(unittest.TestCase):
    def test_reference_failure_stops_generation_with_specific_reason(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / 'work'
            work.mkdir()
            report = WorkflowReport(0, root / 'workflow.log')
            with patch('cuda_agent.workflow.load_task', return_value={'code': '', 'ops': []}), \
                 patch('cuda_agent.workflow.run_command', return_value=(False, 'Reference returned invalid inputs')), \
                 patch('cuda_agent.workflow.generate_cuda') as generate:
                self.assertFalse(execute_workflow(0, WorkflowConfig(work), report, None))
            generate.assert_not_called()
            self.assertIn('invalid inputs', report.failure_reason)

    def test_new_generation_does_not_inherit_old_kernels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / 'work'
            work.mkdir()
            report = WorkflowReport(0, root / 'workflow.log')
            observed = []
            def generate(*args):
                observed.append(list((work / 'kernels').glob('*.cu')))
                (work / 'kernels').mkdir(exist_ok=True)
                (work / 'kernels/old.cu').write_text('old')
                (work / 'model_new.py').write_text('partial')
                return False
            with patch('cuda_agent.workflow.load_task', return_value={'code': '', 'ops': []}), \
                 patch('cuda_agent.workflow.run_command', return_value=(True, 'reference ready')), \
                 patch('cuda_agent.workflow.generate_cuda', side_effect=generate):
                self.assertFalse(execute_workflow(0, WorkflowConfig(work, 2, 1), report, None))
            self.assertEqual(observed, [[], []])

    def test_command_timeout_releases_gpu_lease(self):
        import subprocess
        from cuda_agent.workflow import GPUPool, run_command
        pool = GPUPool(['0'])
        with patch('cuda_agent.workflow.subprocess.run', side_effect=subprocess.TimeoutExpired('command', 1, output=b'partial')):
            success, output = run_command(('python', 'test'), Path('.'), {}, pool)
        self.assertFalse(success)
        self.assertIn('partial', output)
        self.assertEqual(pool._available, ['0'])

    def test_report_links_candidate_and_contains_actual_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = WorkflowReport(6, Path(temporary) / 'workflow.log')
            report.best_candidate = 'correctness/candidates/0002'
            content = report.write(False, 'grid coverage mismatch').read_text()
            self.assertIn('correctness/candidates/0002', content)
            self.assertIn('grid coverage mismatch', content)


if __name__ == "__main__":
    unittest.main()
