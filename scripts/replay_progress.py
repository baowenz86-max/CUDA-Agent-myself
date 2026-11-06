"""Audit historical snapshots offline. Never import generated models or deserialize tensors."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from agent_workdir.utils.checkpoint_trust import checkpoint_mapping, metadata_compatible
from cuda_agent.candidates import compare, progress, trusted_advancement
from cuda_agent.repair import failure_category, repair_effect
from cuda_agent.workflow import SEMANTIC_RULES
from scripts.compare_runs import summarize

TASKS = [84, 87, 99, 114, 122, 125, 136]


def _load_run(root, task):
    reports = sorted((root / f'task_{task:04d}').glob('*/workflow_report.md'))
    if not reports:
        raise ValueError(f'No completed report for task {task}')
    run = reports[-1].parent
    records = []
    for path in sorted((run / 'correctness/candidates').glob('*/record.json')):
        record = json.loads(path.read_text())
        record['snapshot'] = str(path.parent)
        sources = path.parent / 'sources'
        for name, digest in record.get('manifest', {}).items():
            if hashlib.sha256((sources / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Snapshot hash differs: {sources / name}')
        records.append(record)
    return run, records


def _recheck(record, run):
    record = copy.deepcopy(record)
    snapshot = Path(record['snapshot'])
    diagnostic = record.get('diagnostic') or {}
    if not diagnostic:
        return record
    reference = json.loads((snapshot / 'task_contract.json').read_text())
    contract = json.loads((snapshot / 'custom_ops_contract.json').read_text())
    source = (snapshot / 'sources/model_new.py').read_text()
    native = '\n'.join(p.read_text() for p in (snapshot / 'sources/kernels').glob('*') if p.suffix in {'.cpp', '.cu'})
    mapping = checkpoint_mapping(source, reference, contract, native)
    historical_matches = diagnostic.get('matched_checkpoints', 0)
    rows = []
    for index, row in enumerate(mapping):
        row = dict(row)
        # Revalidate correspondence before admitting a historical observation.
        # These are recorded observations, not newly executed tensor comparisons.
        row.update(matched=index < historical_matches, reference=reference['checkpoints'][index].get('value'),
                   observation_origin='historical diagnosis; correspondence rechecked from hashed sources',
                   values_reexecuted=False)
        if index == historical_matches and diagnostic.get('status') == 'failed':
            error = diagnostic.get('error') or {}
            row['mismatch'] = error
            if error.get('actual') and error.get('expected'):
                row['candidate'] = error['actual']
                row['reference'] = error['expected']
                if not metadata_compatible(error['actual'],error['expected']):
                    row.update(trustworthy=False,reason='historical live container/shape/dtype differs')
        elif index >= historical_matches:
            row['matched'] = None
        rows.append(row)
    diagnostic.update(checkpoints=rows, fixture_id=str(run) + ':' + str(reference.get('seed')),
                      reference_sequence=[(r['id'],r['source']) for r in reference['checkpoints']],
                      historical_matched_count=historical_matches)
    diagnostic['matched_checkpoints'] = progress(dict(record, diagnostic=diagnostic))['trusted_matched']
    record['diagnostic'] = diagnostic
    record['progress'] = progress(record)
    return record


def replay(root):
    root = Path(root)
    cases = {}
    for task in TASKS:
        run, original = _load_run(root,task)
        checked = [_recheck(r,run) for r in original]
        by_id = {r['id']:r for r in checked}
        best = None
        decisions = []
        old_useful = new_useful = 0
        for record in checked:
            old_base = by_id.get(record.get('previous_best'))
            old_useful += bool(record.get('decision') == 'rejected' and trusted_advancement(record, old_base))
            accepted, reason = compare(record, best)
            new_useful += bool(not accepted and trusted_advancement(record, best))
            decisions.append({'id':record['id'], 'old_decision':record.get('decision'),
                              'new_decision':'best' if accepted else 'rejected', 'reason':reason,
                              'base_candidate_id':best['id'] if best else None})
            if accepted:
                best = record
        case = {'run':str(run), 'snapshot_hashes_verified':True, 'values_reexecuted':False,
                'selection_decisions':decisions, 'old_useful_candidate_rejections':old_useful,
                'useful_candidate_rollback_count':new_useful,
                'candidates':[{'id':r['id'], 'manifest':r['manifest'], 'progress':progress(r),
                               'checkpoint_mapping':(r.get('diagnostic') or {}).get('checkpoints',[])} for r in checked]}
        if task == 84:
            case.update(passed=failure_category(checked[0])=='environment', classification=failure_category(checked[0]))
        elif task in {87,136}:
            new_id = 3 if task==87 else 4
            accepted,reason = compare(by_id[new_id],by_id[2])
            case.update(passed=accepted and reason=='checkpoint_progress', selected_comparison=[2,new_id], decision_reason=reason)
        elif task in {99,125}:
            selected = by_id[2 if task==99 else 3]
            first = selected['diagnostic']['checkpoints'][0]
            case.update(passed=first['trustworthy'] is False, classification='diagnostic_unreliable', reason=first['reason'])
        elif task == 114:
            diagnostic = checked[0]['diagnostic']
            case.update(passed='不能仅凭' in SEMANTIC_RULES and '输出幅度' in SEMANTIC_RULES,
                        facts={'checkpoint':diagnostic.get('checkpoint'),'error':diagnostic.get('error')},
                        inference='Output magnitude alone does not establish input magnitude; poles and near-pole behavior are mathematical possibilities.',
                        hypothesis='Input-near-pole and implementation error require input-side evidence; neither is established by this replay.')
        elif task == 122:
            effects = [repair_effect(by_id[r['previous_best']],r,r.get('api_analysis')) for r in checked if r.get('previous_best')]
            case.update(passed=all(not e['repair_not_applied'] for e in effects), effects=effects,
                        original_expectation_supported=False,
                        note='Evidence discrepancy: all repair binding hashes changed; candidate 3 changed the topk return type to std::tuple. No repair claims to change the unchanged CUDA file. Historical repair_not_applied is therefore not established; synthetic regression tests verify unapplied claimed target detection separately.')
        cases[str(task)] = case
    baseline = summarize(root,81,140)
    return {'schema_version':2, 'scope':'offline selection/classification audit; no model calls, tensor replay or GPU benchmark',
            'all_passed':all(c['passed'] for c in cases.values()),
            'original_expectations_fully_supported':all(c.get('original_expectation_supported', True) for c in cases.values()),
            'original_expectation_discrepancies':[int(t) for t,c in cases.items() if c.get('original_expectation_supported') is False], 'cases':cases,
            'historical_baseline':baseline,
            'replay_useful_candidate_rollback_count':sum(c['useful_candidate_rollback_count'] for c in cases.values()),
            'old_useful_candidate_rejections_with_rechecked_mapping':sum(c['old_useful_candidate_rejections'] for c in cases.values()),
            'limitation':'Historical matched observations are admitted only after source correspondence checks; tensor values and CUDA correctness were not reexecuted. No success-rate improvement is claimed.'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs',type=Path,default=Path('logs'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    result=replay(args.logs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'all_passed':result['all_passed'], 'original_expectation_discrepancies':result['original_expectation_discrepancies'], 'cases':{k:v['passed'] for k,v in result['cases'].items()}},ensure_ascii=False))
    return 0 if result['all_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
