"""Compare completed task batches without importing torch or calling an API."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from cuda_agent.repair import failure_category, TERMINAL_FAILURES
from cuda_agent.metrics import candidate_metrics, BASELINE_81_140_EVALUABLE, metric_category


def summarize(root, start=11, end=30):
    root = Path(root)
    tasks, missing = {}, []
    failures = Counter()
    api_calls = corrections = api_failures = initial_passes = 0
    repairs = improved_repairs = repair_metadata = 0
    total_seconds = 0.0
    task_metrics = {}
    task_failures = Counter()
    tasks_with_failure_category = Counter()
    for task_id in range(start, end + 1):
        reports = sorted((root / f'task_{task_id:04d}').glob('*/workflow_report.md'))
        if not reports:
            missing.append(task_id)
            continue
        report = reports[-1]
        text = report.read_text()
        status = re.search(r'Status: \*\*(.*?)\*\*', text)
        if not status:
            missing.append(task_id)
            continue
        duration = re.search(r'Duration: `([\d.]+)s`', text)
        total_seconds += float(duration.group(1)) if duration else 0
        records = [json.loads(p.read_text()) for p in sorted((report.parent / 'correctness/candidates').glob('*/record.json'))]
        metrics_path = report.parent / 'workflow_report.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else candidate_metrics(records)
        if not metrics_path.exists():
            metrics['profiling_failed'] = 'Correctness passed, but profiling failed' in text
            if not records:
                category = failure_category({'stage': 'reference' if 'Reference preflight failed' in text else 'unknown', 'error': text})
            else:
                category = failure_category(next((r for r in records if r.get('id') == records[-1].get('best_candidate')), records[-1]))
            metrics['failure_type'] = category if status.group(1) != 'SUCCESS' else ''
        task_metrics[str(task_id)] = metrics
        if metrics.get('failure_type'):
            task_failures[metric_category(metrics['failure_type'])] += 1
        observed_categories = set(metrics.get('candidate_failures', {}))
        if metrics.get('failure_type'):
            observed_categories.add(metric_category(metrics['failure_type']))
        tasks_with_failure_category.update(observed_categories)
        initial_passes += bool(metrics.get('first_pass'))
        for record in records:
            if not record.get('success'):
                category = ('generation' if record.get('error', '').startswith('Generation failed') else failure_category(record))
                failures[metric_category(category)] += 1
            if record.get('api_stage') is not None:
                repair_metadata += 1
                if record['api_stage'] == 'repair':
                    repairs += 1
                    improved_repairs += record.get('decision') == 'best'
        api_path = report.parent / 'correctness/api_calls.jsonl'
        if api_path.exists():
            for line in api_path.read_text().splitlines():
                event = json.loads(line)
                api_calls += 1
                api_failures += event.get('success') is False
                corrections += event.get('correction_attempt', 0) > 0
        tasks[str(task_id)] = {'status': status.group(1), 'run': str(report.parent)}
    count = len(tasks)
    successful = sum(x['status'] == 'SUCCESS' for x in tasks.values())
    result = {'tasks': tasks, 'missing_tasks': missing, 'completed_tasks': count,
            'successful_tasks': successful, 'success_rate': successful / count if count else None,
            'first_candidate_pass_rate': initial_passes / count if count else None,
            'candidate_failures': dict(failures), 'api_calls': api_calls,
            'api_failed_calls': api_failures, 'format_correction_calls': corrections,
            'evaluated_repairs': repairs if repair_metadata else None,
            'effective_repair_rate': improved_repairs / repairs if repairs else None,
            'summed_task_seconds': round(total_seconds, 2)}
    entered = [m for m in task_metrics.values() if not m.get('first_pass') and m.get('entered_repair')]
    correctness = sum(m.get('correctness_success', False) for m in task_metrics.values())
    evaluable = [int(t) for t, m in task_metrics.items() if m.get('failure_type') not in TERMINAL_FAILURES]
    fixed_ids = [t for t in BASELINE_81_140_EVALUABLE if start <= t <= end]
    def rate(ids):
        return sum(task_metrics.get(str(i), {}).get('correctness_success', False) for i in ids) / len(ids) if ids else None
    result.update(schema_version=2, task_metrics=task_metrics, task_failures=dict(task_failures),
                  tasks_with_failure_category=dict(tasks_with_failure_category),
                  requested_tasks=end-start+1, correctness_successes=correctness,
                  first_pass_rate=initial_passes/(end-start+1), workflow_success_rate=successful/(end-start+1),
                  final_correctness_rate=correctness/(end-start+1),
                  profiling_failed_tasks=sum(bool(m.get('profiling_failed')) for m in task_metrics.values()),
                  repair_recovery_rate=sum(bool(m.get('repair_recovered')) for m in entered)/len(entered) if entered else None,
                  repair_recovery_denominator=len(entered),
                  average_repair_attempts=sum(m.get('repair_attempts', 0) for m in task_metrics.values())/count if count else None,
                  baseline_fixed_evaluable_ids=fixed_ids, baseline_fixed_evaluable_count=len(fixed_ids),
                  baseline_fixed_correctness_rate=rate(fixed_ids),
                  new_evaluable_ids=evaluable, new_evaluable_count=len(evaluable), new_evaluable_correctness_rate=rate(evaluable))
    for key in ('progression_accepted', 'same_failure', 'repair_not_applied', 'untrusted_checkpoints',
                'useful_candidate_rollback_count', 'correctness_finishing_repairs'):
        values = [m.get(key) for m in task_metrics.values()]
        result[key] = sum(values) if values and all(v is not None for v in values) else None
    attempts = sum(m.get('repair_attempts', 0) for m in task_metrics.values())
    result['successful_repair_rate'] = result['correctness_finishing_repairs']/attempts if result['correctness_finishing_repairs'] is not None and attempts else None
    rates = [m for m in task_metrics.values() if m.get('repair_attempts')]
    result['progression_effective_rate'] = sum(m['progression_effective_rate']*m['repair_attempts'] for m in rates)/attempts if rates and all(m.get('progression_effective_rate') is not None for m in rates) else None
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--task-start', type=int, default=11)
    parser.add_argument('--task-end', type=int, default=30)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.task_start < 0 or args.task_end < args.task_start:
        parser.error('task range must be nonnegative and inclusive')
    result = {name: summarize(root, args.task_start, args.task_end)
              for name, root in [('before', args.before), ('after', args.after)]}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
