"""Compare completed task batches without importing torch or calling an API."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from cuda_agent.repair import failure_category


def summarize(root, start=11, end=30):
    root = Path(root)
    tasks, missing = {}, []
    failures = Counter()
    api_calls = corrections = api_failures = initial_passes = 0
    repairs = improved_repairs = repair_metadata = 0
    total_seconds = 0.0
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
        initial_passes += bool(records and records[0].get('success'))
        for record in records:
            if not record.get('success'):
                category = ('generation' if record.get('error', '').startswith('Generation failed') else failure_category(record))
                failures[category] += 1
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
    return {'tasks': tasks, 'missing_tasks': missing, 'completed_tasks': count,
            'successful_tasks': successful, 'success_rate': successful / count if count else None,
            'first_candidate_pass_rate': initial_passes / count if count else None,
            'candidate_failures': dict(failures), 'api_calls': api_calls,
            'api_failed_calls': api_failures, 'format_correction_calls': corrections,
            'evaluated_repairs': repairs if repair_metadata else None,
            'effective_repair_rate': improved_repairs / repairs if repairs else None,
            'summed_task_seconds': round(total_seconds, 2)}


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
