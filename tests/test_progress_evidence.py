"""Offline regressions for trusted progress, native interfaces and repair accounting."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from cuda_agent.candidates import better, compare, progress, CandidateStore
from cuda_agent.repair import error_fingerprint, failure_category, repair_effect
from cuda_agent.metrics import candidate_metrics, BASELINE_81_140_EVALUABLE
from cuda_agent.validation import coverage_warnings
from cuda_agent.workflow import execute_workflow, WorkflowConfig
from cuda_agent.report import WorkflowReport
from cuda_agent.api import _validate_analysis
from agent_workdir.utils.checkpoint_trust import checkpoint_mapping, metadata_compatible
from agent_workdir.utils.operator_contract import (validate_contract, cpp_parameter_type, pybind_argument,
    source_binding_checks, binding_contract_errors, model_interface_errors)
from agent_workdir.utils.dependencies import discover_cudnn, needs_cudnn, configure_dependency
from tests.test_correctness import verification
from tests.test_agent_repairs import contract, response
from tests import test_agent_repairs as repair_tests


def candidate(matched=0, ratio=.5, status='failed'):
    checks = [{'id': f'step_{i:03}', 'trustworthy': True, 'matched': True} for i in range(matched)]
    checks.append({'id': f'step_{matched:03}', 'trustworthy': True, 'matched': False,
                   'reference': {'shape': [3]}, 'mismatch': {'mismatch_ratio': .5}})
    return {'stage': 'verification', 'success': False, 'verification': verification(ratio),
            'diagnostic': {'status': status, 'input_seed': 2027, 'fixture_id': 'fixture',
                           'reference_sequence': ['a','b','c','d'], 'checkpoints': checks}}


class ProgressEvidenceTests(unittest.TestCase):
    def test_checkpoint_progress_precedes_higher_final_mismatch(self):
        self.assertEqual(compare(candidate(2,.9), candidate(0,.1)), (True, 'checkpoint_progress'))

    def test_compile_regression_cannot_replace_verified_candidate(self):
        self.assertFalse(better({'stage':'compile', 'error':'syntax'}, candidate(2)))

    def test_unavailable_preserves_partial_trusted_progress(self):
        self.assertTrue(better(candidate(2,.9,'unavailable'), candidate(0,.1,'unavailable')))

    def test_missing_trust_is_not_inferred_from_legacy_matched_count(self):
        old, new = candidate(), candidate()
        new['diagnostic'] = dict(old['diagnostic'], matched_checkpoints=999, checkpoints=[])
        self.assertFalse(better(new, old))

    def test_missing_diagnostic_neutral_but_confirmed_regression_wins(self):
        old, new = candidate(2,.5), candidate(0,.1)
        self.assertEqual(compare(new,old)[1], 'checkpoint_regression')
        new['diagnostic'] = {'status':'unavailable'}
        self.assertTrue(better(new,old))

    def test_fixture_seed_and_reference_sequence_must_match(self):
        for key,value in [('fixture_id','other'),('input_seed',4),('reference_sequence',['other'])]:
            old,new=candidate(),candidate(2,.1)
            new['diagnostic'][key]=value
            self.assertFalse(better(new,old),key)

    def test_normal_seed_missing_observation_and_pass_regression(self):
        old=candidate()
        for mutation in ('seed','missing','passed'):
            new=candidate(2,.1)
            if mutation=='seed': new['verification']['checks'][0]['seed']=999
            if mutation=='missing': new['verification']['checks'].pop()
            if mutation=='passed':
                old['verification']['checks'][0]['status']='passed'
                new['verification']['checks'][0]['status']='failed'
            self.assertFalse(better(new,old))

    def test_same_checkpoint_local_mismatch_precedes_final(self):
        old,new=candidate(1,.1),candidate(1,.9)
        new['diagnostic']['checkpoints'][1]['mismatch']['mismatch_ratio']=.2
        self.assertEqual(compare(new,old),(True,'checkpoint_mismatch_progress'))

    def test_only_five_normal_passes_establish_success(self):
        c=candidate();c['success']=True
        c['verification']=verification(.0,5);c['verification']['status']='passed'
        self.assertTrue(progress(c)['correctness'])
        c['verification']['checks'].pop()
        self.assertFalse(progress(c)['correctness'])

    def test_rollback_restores_task_contract_and_keeps_effect_history(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp)/'work';run=Path(temp)/'run';work.mkdir();run.mkdir()
            (work/'model_new.py').write_text('base');(run/'task_contract.json').write_text('{"v":1}')
            store=CandidateStore(work,run);c=candidate(2)
            base=store.capture('verification',False,'bad',c['verification'],c['diagnostic'])
            store.restore_best();store.repair_base=base
            (work/'model_new.py').write_text('regression');(run/'task_contract.json').write_text('{"v":2}')
            c=candidate(0,.1);store.capture('verification',False,'bad',c['verification'],c['diagnostic'])
            effects=(run/'repair_effects.jsonl').read_bytes();store.restore_best()
            self.assertEqual((run/'task_contract.json').read_text(),'{"v":1}')
            self.assertEqual((run/'repair_effects.jsonl').read_bytes(),effects)


class RepairEffectTests(unittest.TestCase):
    def test_same_failure_and_unapplied_are_independent(self):
        base=candidate();base.update(id=1,manifest={'kernels/custom_ops.cu':'a','kernels/binding.cpp':'b'})
        new=copy.deepcopy(base);new.update(id=2,manifest={'kernels/custom_ops.cu':'a','kernels/binding.cpp':'c'})
        effect=repair_effect(base,new,{'target_files':['kernels/custom_ops.cu']})
        self.assertEqual(effect['outcome'],'same_failure');self.assertTrue(effect['repair_not_applied'])
        effect=repair_effect(base,new,{'target_files':['kernels/binding.cpp']})
        self.assertFalse(effect['repair_not_applied'])

    def test_same_fingerprint_with_progress_is_progressed(self):
        base,new=candidate(1),candidate(1)
        new['diagnostic']['checkpoints'][1]['mismatch']['mismatch_ratio']=.1
        self.assertEqual(error_fingerprint(base),error_fingerprint(new))
        self.assertEqual(repair_effect(base,new)['outcome'],'progressed')

    def test_fingerprint_normalizes_paths_and_addresses_preserves_dimensions(self):
        a={'stage':'compile','error':'TypeError: /tmp/run1/a.cpp pointer 0x1234 shape (32, 64)'}
        b={'stage':'compile','error':'TypeError: /other/run2/a.cpp pointer 0x9876 shape (32, 64)'}
        self.assertEqual(error_fingerprint(a),error_fingerprint(b))
        b['error']=b['error'].replace('64','65');self.assertNotEqual(error_fingerprint(a),error_fingerprint(b))

    def test_noop_retries_are_bounded_and_emit_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'model_new.py').write_text('same')
            value=response(json.dumps({'analysis':{'hypothesis':'test','evidence':['observed'], 'changes':['modify']},'files':{'model_new.py':'same'}}))
            result,requests,events=repair_tests.APIRepairTests().run_mock(root,[value]*3,analysis=True)
            self.assertFalse(result);self.assertEqual(len(requests),3)
            effects=[json.loads(x) for x in (root/'run/repair_effects.jsonl').read_text().splitlines()]
            self.assertEqual(len(effects),3);self.assertTrue(all(x['repair_not_applied'] for x in effects))
            self.assertTrue(all(x['outcome']=='same_failure' for x in effects))

    def test_target_files_must_be_generated_source_paths(self):
        analysis={'hypothesis':'test','evidence':['fact'],'changes':['change'],'target_files':['../model.py']}
        with self.assertRaises(ValueError): _validate_analysis({'analysis':analysis},True)


class TrustTests(unittest.TestCase):
    def mapping(self, source, stages=None, native='float sigmoid_forward(float x){return 1.0f/(1.0f+expf(-x));}'):
        reference={'checkpoints':stages or [{'id':'step_001','source':'x = torch.sigmoid(x)'}, {'id':'step_002','source':'return x'}]}
        return checkpoint_mapping(source,reference,native_source=native)

    def test_straight_line_mapping_and_pure_helper(self):
        source='''class ModelNew:
 def helper(self,x): return cuda_extension.sigmoid_forward(x)
 def forward(self,x):
  x=self.helper(x)
  return x
 def debug_forward(self,x):
  x=self.helper(x)
  yield 'step_001',x
  yield 'step_002',x
'''
        self.assertTrue(all(r['trustworthy'] for r in self.mapping(source)))

    def test_yield_input_and_fused_operation_are_untrusted(self):
        for expression in ['x', 'cuda_extension.fused_sqrt_round(x)']:
            source=f'''class ModelNew:
 def forward(self,x):
  x=cuda_extension.sigmoid_forward(x)
  return x
 def debug_forward(self,x):
  x={expression}
  yield 'step_001',x
  yield 'step_002',x
'''
            self.assertFalse(self.mapping(source)[0]['trustworthy'])

    def test_later_missing_step_preserves_verified_prefix(self):
        source='''class ModelNew:
 def forward(self,x):
  x=cuda_extension.sigmoid_forward(x)
  return x
 def debug_forward(self,x):
  x=cuda_extension.sigmoid_forward(x)
  yield 'step_001',x
'''
        rows=self.mapping(source)
        self.assertTrue(rows[0]['trustworthy']);self.assertFalse(rows[1]['trustworthy'])

    def test_matching_label_or_shape_does_not_establish_semantic_mapping(self):
        source='''class ModelNew:
 def forward(self,x):
  x=cuda_extension.fused_sqrt_round(x)
  return x
 def debug_forward(self,x):
  x=cuda_extension.fused_sqrt_round(x)
  yield 'step_001',x
  yield 'step_002',x
'''
        rows=self.mapping(source,[{'id':'step_001','source':'x = torch.sqrt(x)'},{'id':'step_002','source':'return x'}], 'void fused_sqrt_round();')
        self.assertFalse(rows[0]['trustworthy'])

    def test_shape_dtype_and_container_use_live_metadata(self):
        a={'type':'tensor','shape':[2,3],'dtype':'float32'}
        self.assertTrue(metadata_compatible(a,dict(a)))
        for b in [dict(a,shape=[4,3]),dict(a,dtype='float16'),{'type':'tuple','items':[a]}]:
            self.assertFalse(metadata_compatible(a,b))
        # Another input may legitimately have another shape on both sides.
        self.assertTrue(metadata_compatible(dict(a,shape=[4,3]),dict(a,shape=[4,3])))


class InterfaceTests(unittest.TestCase):
    def test_four_required_optional_nullable_combinations(self):
        for optional in (False,True):
            for nullable in (False,True):
                c=contract();p=c['operators'][0]['parameters'][1]
                p.update(kind='int',optional=optional,nullable=nullable,required=not optional)
                p.pop('default',None)
                if optional: p['default']=None if nullable else 7
                validate_contract(c)
                self.assertEqual('optional' in cpp_parameter_type(p),nullable)
                self.assertEqual(' = ' in pybind_argument(p),optional)
                if nullable and not optional:
                    self.assertEqual(pybind_argument(p),'py::arg("bias")')

    def test_required_default_and_contradictory_required_rejected(self):
        for mutation in ('default','required'):
            c=contract();p=c['operators'][0]['parameters'][0]
            p[mutation]=1 if mutation=='default' else False
            with self.assertRaises(ValueError): validate_contract(c)

    def test_static_binding_parameter_order_default_and_nullable(self):
        c=contract()
        good='torch::Tensor op(torch::Tensor x, c10::optional<torch::Tensor> bias) {return x;} void reg(){m.def("op", &op, py::arg("x"), py::arg("bias") = py::none());}'
        self.assertEqual(source_binding_checks(good,c)['status'],'passed')
        for bad in [good.replace('c10::optional<torch::Tensor>','torch::Tensor'),good.replace('arg("x")','arg("wrong")'),good.replace(' = py::none()','')]:
            self.assertTrue(source_binding_checks(bad,c)['errors'])
        self.assertEqual(source_binding_checks('DYNAMIC_BINDING(op)',c)['status'],'pending')

    def test_compiled_nullable_checked_in_both_directions(self):
        def op(*args): raise AssertionError('native execution forbidden')
        ext=types.SimpleNamespace(op=op);c=contract();c['operators'][0]['parameters'][0]['nullable']=False
        op.__doc__='op(x: torch.Tensor | None, bias: torch.Tensor | None = None) -> torch.Tensor'
        self.assertTrue(binding_contract_errors(ext,c))
        op.__doc__='op(x: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor'
        self.assertTrue(binding_contract_errors(ext,c))

    def test_signature_and_state_keys_checked_without_forward(self):
        class Reference:
            def __init__(self, n=2): pass
            def forward(self,x): raise AssertionError('forward executed')
            def state_dict(self): return {'weight':types.SimpleNamespace(shape=(2,))}
        class Good(Reference): pass
        class Bad(Reference):
            def forward(self,x,extra=2): raise AssertionError('forward executed')
            def state_dict(self): return {'wrong':types.SimpleNamespace(shape=(2,))}
        self.assertEqual(model_interface_errors(Reference(),Good()),[])
        self.assertEqual(len(model_interface_errors(Reference(),Bad())),2)


class DependencyRoutingTests(unittest.TestCase):
    def test_reference_conv_detection_uses_imports_not_ops_labels(self):
        self.assertTrue(needs_cudnn('from torch.nn import ConvTranspose2d as C\nx=C(3,4,2)'))
        self.assertFalse(needs_cudnn('x="Conv2d"\nx=torch.sigmoid(x)'))

    def test_cudnn_discovery_requires_header_and_linkable_library(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'include').mkdir();(root/'lib').mkdir()
            env={'CUDNN_HOME':str(root)}
            with patch('agent_workdir.utils.dependencies._linkable',return_value=(True,'')) as linker:
                self.assertEqual(discover_cudnn(env,roots=[])['failure_type'],'environment')
                (root/'include/cudnn.h').write_text('header')
                self.assertEqual(discover_cudnn(env,roots=[])['status'],'failed')
                (root/'lib/libcudnn.so.9').write_text('library')
                dependency=discover_cudnn(env,roots=[])
                self.assertEqual(dependency['status'],'passed');linker.assert_called_once()
                configure_dependency(dependency,env)
                self.assertIn(str(root/'include'),env['CPATH'])
                old=dependency['fingerprint'];(root/'include/cudnn.h').write_text('changed header')
                self.assertNotEqual(discover_cudnn(env,roots=[])['fingerprint'],old)
            with patch('agent_workdir.utils.dependencies._linkable',return_value=(False,'broken')):
                self.assertEqual(discover_cudnn(env,roots=[])['status'],'failed')

    def test_python_nvidia_package_and_torch_directory_are_discovered(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            for suffix in ('nvidia/cudnn','torch'):
                root=base/suffix;(root/'include').mkdir(parents=True);(root/'lib').mkdir()
                (root/'include/cudnn.h').write_text('header');(root/'lib/libcudnn.so').write_text('library')
            with patch('agent_workdir.utils.dependencies.site.getsitepackages',return_value=[temp]), patch('agent_workdir.utils.dependencies.site.getusersitepackages',return_value=temp), patch('agent_workdir.utils.dependencies.importlib.util.find_spec',return_value=None), patch('agent_workdir.utils.dependencies._linkable',return_value=(True,'')):
                self.assertEqual(discover_cudnn({})['status'],'passed')

    def test_terminal_failures_never_call_repair(self):
        for kind,message in [('environment','fatal error: cudnn.h: No such file or directory'),('resource','CUDA out of memory'),('reference','reference failed')]:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);work=root/'work';work.mkdir();run=root/'correctness'
                report=WorkflowReport(0,root/'workflow.log')
                def generate(*args): (work/'model_new.py').write_text('generated');return True
                def command(argv,*args):
                    if argv[-1]=='utils.preflight': return True,'ready'
                    if argv[-1]=='utils/compile.sh': return kind=='reference',message
                    if argv[-1]=='--compiled': return True,'ready'
                    if argv[-1]=='utils.verification':
                        v={'status':'failed','checks':[{'index':0,'seed':2027,'status':'failed','stage':'reference','error':{'kind':'runtime','message':message}}]}
                        (run/'verification.json').write_text(json.dumps(v));return False,message
                    raise AssertionError(argv)
                with patch('cuda_agent.workflow.load_task',return_value={'code':'','ops':[]}),patch('cuda_agent.workflow.generate_cuda',side_effect=generate),patch('cuda_agent.workflow.send_feedback') as repair,patch('cuda_agent.workflow.run_command',side_effect=command),patch('cuda_agent.workflow.check_sources',return_value=types.SimpleNamespace(errors=[],warnings=[],as_dict=lambda:{})):
                    self.assertFalse(execute_workflow(0,WorkflowConfig(work),report,None))
                    repair.assert_not_called();self.assertEqual(report.repairs_used,0);self.assertEqual(report.failure_type,kind)

    def test_device_transfer_clones_only_shared_storage(self):
        tree=ast.parse(Path('agent_workdir/utils/verification.py').read_text())
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='device_inputs')
        namespace={'transform_tree':lambda values,fn:[fn(v) for v in values]}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<device_inputs>','exec'),namespace)
        class Tensor:
            def __init__(self, pointer, moved=None): self.pointer=pointer;self.moved=moved;self.clones=0
            def to(self,device): return self.moved or self
            def data_ptr(self): return self.pointer
            def clone(self): self.clones+=1;return Tensor(self.pointer+1)
        same=Tensor(1);independent=Tensor(3);cross=Tensor(2,independent);alias=Tensor(4,Tensor(4))
        result=namespace['device_inputs']([same,cross,alias],'cuda')
        self.assertEqual(same.clones,1);self.assertEqual(independent.clones,0);self.assertIs(result[1],independent)
        self.assertEqual(alias.moved.clones,1)


class CoverageMetricTests(unittest.TestCase):
    def test_unrelated_loop_cannot_hide_another_kernel_coverage(self):
        source='''__global__ void bad(float* x,int n) {int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)x[i]=0;}
__global__ void good(float* x,int n) {for(int i=threadIdx.x;i<n;i+=blockDim.x*gridDim.x)x[i]=0;}
void launch(){int n=1000; bad<<<2,32>>>(x,n);good<<<2,32>>>(x,n);}'''
        warnings=coverage_warnings(source)
        self.assertEqual(len(warnings),1);self.assertIn('numel=1000',warnings[0]);self.assertIn('bad',warnings[0])

    def test_unknown_extent_warns_as_risk_only(self):
        source='__global__ void k(){int i=threadIdx.x;} void launch(){int grid=100000;if(grid>65535)grid=65535;k<<<grid,256>>>(x,n);}'
        self.assertIn('possible',coverage_warnings(source)[0])

    def test_metrics_audit_rejected_trusted_progress_and_old_unknowns(self):
        old,new=candidate(0),candidate(2)
        old.update(id=1,schema_version=2,decision='best',previous_best=None)
        new.update(id=2,schema_version=2,decision='rejected',previous_best=1)
        self.assertEqual(candidate_metrics([old,new])['useful_candidate_rollback_count'],1)
        self.assertIsNone(candidate_metrics([{'id':1,'stage':'compile'}])['same_failure'])
        self.assertEqual(len(BASELINE_81_140_EVALUABLE),55)

    def test_profiling_failure_is_separate_from_correctness(self):
        with tempfile.TemporaryDirectory() as temp:
            report=WorkflowReport(0,Path(temp)/'workflow.log',correctness_passed=True,profiling_failed=True)
            report.write(False)
            metrics=json.loads((Path(temp)/'workflow_report.json').read_text())
            self.assertTrue(metrics['correctness_success']);self.assertTrue(metrics['profiling_failed']);self.assertFalse(metrics['workflow_success'])


class IntegrationEvidenceTests(unittest.TestCase):
    def test_diagnostic_exception_retains_two_trusted_observations_without_torch(self):
        import hashlib
        import inspect
        from contextlib import nullcontext
        from agent_workdir.utils.diagnostics import ComparisonFailure
        tree=ast.parse(Path('agent_workdir/utils/verification.py').read_text())
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='diagnose')
        stages=[{'id':f'step_{i:03}','source':'x = op(x)'} for i in range(3)]
        class Model:
            def load_state_dict(self,*args,**kwargs): pass
            def debug_forward(self,*inputs):
                yield 'step_000',1
                yield 'step_001',2
                raise RuntimeError('debug interrupted')
        with tempfile.TemporaryDirectory() as temp:
            namespace={'__package__':'agent_workdir.utils','Path':Path,'inspect':inspect,'hashlib':hashlib,'json':json,
                'fixture_directory':lambda:Path(temp),'reference_trace':lambda m:(lambda *args:iter([(s['id'],i+1) for i,s in enumerate(stages)]),stages),
                'seed_all':lambda s:None,'rng_state':lambda:{},'restore_rng':lambda s:None,
                'device_inputs':lambda i,d:i,'block_torch_functional':nullcontext,'synchronize':lambda d:None,
                'check_equal':lambda a,b,p:None,'ComparisonFailure':ComparisonFailure,'traceback':__import__('traceback')}
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'<diagnose>','exec'),namespace)
            mapping=[dict(s,trustworthy=True,reason='checked') for s in stages]
            with patch('agent_workdir.utils.checkpoint_trust.checkpoint_mapping',return_value=mapping),patch('agent_workdir.utils.diagnostics.describe',return_value={'type':'tensor','shape':[2],'dtype':'float32'}),patch('agent_workdir.utils.diagnostics.read_json',side_effect=lambda path,default: {'checkpoints':stages} if path.name=='task_contract.json' else {}):
                result=namespace['diagnose'](Model(),Model(),{'state_dict':{}},[0],2027,'cpu')
            self.assertEqual(result['status'],'unavailable')
            self.assertEqual(result['matched_checkpoints'],2)
            self.assertEqual(len(result['checkpoints']),2)

    def test_identity_stub_does_not_gain_trust_from_its_name(self):
        from agent_workdir.utils.checkpoint_trust import native_correspondence
        self.assertFalse(native_correspondence('sigmoid_forward','sigmoid','float sigmoid_forward(float x){return x;}')['supported'])
        self.assertFalse(native_correspondence('sigmoid_forward','sigmoid','float sigmoid_forward(float x);')['supported'])

    def test_grid_product_outside_loop_is_not_a_grid_stride_loop(self):
        code='__global__ void k(float*x,int n){int stride=blockDim.x*gridDim.x;for(int i=0;i<n;i+=1)x[i]=0;} void launch(){int n=1000;k<<<2,32>>>(x,n);}'
        self.assertEqual(len(coverage_warnings(code)),1)

    def test_precompile_interface_failure_stops_native_compilation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);work=root/'work';work.mkdir();run=root/'correctness'
            report=WorkflowReport(0,root/'workflow.log');commands=[]
            def command(argv,*args):
                commands.append(argv)
                if argv[-1]=='utils.preflight':
                    (run/'task_contract.json').write_text('{}');return True,'reference ready'
                if argv[-1]=='--interface-only':
                    (run/'interface_precompile.json').write_text('{"failure_type":"interface"}')
                    return False,'state_dict mismatch'
                raise AssertionError('native compilation must not run')
            def generate(*args): (work/'model_new.py').write_text('generated');return True
            with patch('cuda_agent.workflow.load_task',return_value={'code':'','ops':[]}),patch('cuda_agent.workflow.generate_cuda',side_effect=generate),patch('cuda_agent.workflow.run_command',side_effect=command),patch('cuda_agent.workflow.check_sources',return_value=types.SimpleNamespace(errors=[],warnings=[],as_dict=lambda:{})):
                self.assertFalse(execute_workflow(0,WorkflowConfig(work,repair_attempts=0),report,None))
            self.assertEqual(report.failure_type,'interface')
            self.assertEqual(len(commands),2)

    def test_preflight_missing_torch_is_environment_before_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);work=root/'work';work.mkdir();report=WorkflowReport(0,root/'workflow.log')
            with patch('cuda_agent.workflow.load_task',return_value={'code':'','ops':[]}),patch('cuda_agent.workflow.generate_cuda') as generation,patch('cuda_agent.workflow.send_feedback') as repair,patch('cuda_agent.workflow.run_command',return_value=(False,"ModuleNotFoundError: No module named 'torch'")):
                self.assertFalse(execute_workflow(0,WorkflowConfig(work),report,None))
                generation.assert_not_called();repair.assert_not_called()
            self.assertEqual(report.failure_type,'environment')

    def test_terminal_evaluation_after_applied_repair_does_not_consume_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);work=root/'work';work.mkdir();report=WorkflowReport(0,root/'workflow.log');compiles=[]
            def command(argv,*args):
                if argv[-1]=='utils.preflight': return True,'ready'
                if argv[-1]=='utils/compile.sh':
                    compiles.append(1)
                    return False,'syntax error' if len(compiles)==1 else 'CUDA out of memory'
                raise AssertionError(argv)
            def generate(*args): (work/'model_new.py').write_text('original');return True
            def feedback(*args): (work/'model_new.py').write_text('repaired');return True
            with patch('cuda_agent.workflow.load_task',return_value={'code':'','ops':[]}),patch('cuda_agent.workflow.generate_cuda',side_effect=generate),patch('cuda_agent.workflow.send_feedback',side_effect=feedback) as repair,patch('cuda_agent.workflow.run_command',side_effect=command),patch('cuda_agent.workflow.check_sources',return_value=types.SimpleNamespace(errors=[],warnings=[],as_dict=lambda:{})):
                self.assertFalse(execute_workflow(0,WorkflowConfig(work),report,None))
                repair.assert_called_once()
            self.assertEqual(report.repairs_used,0);self.assertEqual(report.failure_type,'resource')

    def test_new_empty_reports_have_zero_counts_and_legacy_reports_unknown(self):
        self.assertEqual(candidate_metrics([],complete=True)['same_failure'],0)
        self.assertIsNone(candidate_metrics([])['same_failure'])

    def test_same_compile_root_failure_ignores_build_paths_and_line_numbers(self):
        a={'stage':'compile','error':'[1/4] compile /tmp/a/model.cpp\n/tmp/a/binding.cpp:50:3: error: wrong type (32, 64)'}
        b={'stage':'compile','error':'[2/4] compile /tmp/b/other.cpp\n/tmp/b/binding.cpp:70:9: error: wrong type (32, 64)'}
        self.assertEqual(error_fingerprint(a),error_fingerprint(b))

    def test_recovery_rate_uses_only_initial_failures_that_entered_repair(self):
        from scripts.compare_runs import summarize
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for task,first,recovered in [(1,True,False),(2,False,True),(3,False,False)]:
                run=root/f'task_{task:04}'/'run';run.mkdir(parents=True)
                (run/'workflow_report.md').write_text('Status: **SUCCESS**\n')
                data={'schema_version':2,'first_pass':first,'correctness_success':first or recovered,
                      'entered_repair':recovered,'repair_recovered':recovered,'failure_type':'', 'repair_attempts':int(recovered)}
                (run/'workflow_report.json').write_text(json.dumps(data))
            result=summarize(root,1,3)
            self.assertEqual(result['repair_recovery_denominator'],1)
            self.assertEqual(result['repair_recovery_rate'],1)
            self.assertEqual(result['first_pass_rate'],1/3)

    def test_unreliable_diagnosis_adds_explicit_routing_guidance(self):
        from cuda_agent.workflow import send_feedback
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'diagnostic.json').write_text('{"failure_type":"diagnostic_unreliable"}')
            with patch('cuda_agent.workflow.run_agent',return_value=True) as run:
                send_feedback('mismatch',root,'numerical',root)
            self.assertIn('禁止根据该检查点断言根因',run.call_args.args[0])

    def test_unsafe_later_helper_does_not_erase_independently_valid_prefix(self):
        source='''class ModelNew:
 def unsafe(self,x):
  for i in range(3): x=x+1
  return x
 def forward(self,x):
  x=cuda_extension.sigmoid_forward(x)
  x=self.unsafe(x)
  return x
 def debug_forward(self,x):
  x=cuda_extension.sigmoid_forward(x)
  yield 'step_001',x
  x=self.unsafe(x)
  yield 'step_002',x
  yield 'step_003',x
'''
        reference={'checkpoints':[{'id':'step_001','source':'x = torch.sigmoid(x)'},{'id':'step_002','source':'x = torch.round(x)'},{'id':'step_003','source':'return x'}]}
        mapping=checkpoint_mapping(source,reference,native_source='float sigmoid_forward(float x){return 1/(1+expf(-x));}')
        self.assertTrue(mapping[0]['trustworthy']);self.assertFalse(mapping[1]['trustworthy'])

    def test_untrusted_live_shape_is_counted_even_when_static_mapping_matches(self):
        c=candidate();c.update(id=1,schema_version=2)
        c['diagnostic']['mapping']=[{'id':'step_000','trustworthy':True}]
        c['diagnostic']['checkpoints'][0]['trustworthy']=False
        self.assertEqual(candidate_metrics([c])['untrusted_checkpoints'],1)


    def test_binding_and_contract_failures_have_specific_routes(self):
        self.assertEqual(failure_category({'stage':'verification','error':'TypeError: incompatible function arguments'}),'binding')
        self.assertEqual(failure_category({'stage':'sources','error':'op.bias: C++ type/nullable differs'}),'contract')


    def test_sqrt_label_does_not_hide_visible_native_round_fusion(self):
        from agent_workdir.utils.checkpoint_trust import native_correspondence
        evidence=native_correspondence('sqrt_forward','sqrt','float sqrt_forward(float x){return roundf(sqrtf(x));}')
        self.assertFalse(evidence['supported'])
