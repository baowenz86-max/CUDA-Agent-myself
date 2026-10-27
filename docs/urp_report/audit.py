"""Read-only audit of local task reports and archived artifacts. Run from repo root."""
from pathlib import Path
from collections import Counter
import json
import re

ROOT = Path(__file__).resolve().parents[2]
reports = []
missing = []
for task in range(50):
    paths = sorted((ROOT / f'logs/task_{task:04d}').glob('*/workflow_report.md'))
    if not paths:
        missing.append(task)
        continue
    path = paths[-1]
    text = path.read_text()
    match = re.search(r'Status: \*\*(.*?)\*\*', text)
    reports.append({'task': task, 'report': str(path.relative_to(ROOT)),
                    'status': match.group(1) if match else 'UNKNOWN',
                    'failure_summary': text.split('## Failure Summary')[-1].split('## Reproduction')[0].strip() if '## Failure Summary' in text else ''})
archives = []
for path in sorted((ROOT / 'results').glob('task_*')):
    artifacts = {suffix: [str(p.relative_to(ROOT)) for p in path.rglob('*' + suffix) if p.is_file() and p.stat().st_size] for suffix in ['.cu', '.ptx', '.cubin', '.sass']}
    log = path / 'verification.log'
    verified = log.exists() and '[PASS] verify success' in log.read_text()
    complete = all(artifacts.values())
    archives.append({'task': path.name, 'artifacts': artifacts, 'complete': complete,
                     'verification_pass_marker': verified, 'eligible_for_kernel_audit': complete and verified})
output = {'scope': 'task 0000–0049, latest completed report per task; historical results separately',
          'report_counts': dict(Counter(r['status'] for r in reports)),
          'tasks_without_completed_report': missing, 'reports': reports, 'archives': archives,
          'archive_counts': {'total': len(archives), 'complete': sum(a['complete'] for a in archives),
                             'complete_and_verified': sum(a['eligible_for_kernel_audit'] for a in archives)},
          'limitations': 'Presence and verification markers only; no hash lineage, ABI, symbol mapping or kernel execution audit.'}
(ROOT / 'docs/urp_report/evidence.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
print(json.dumps({'reports': output['report_counts'], 'archives': output['archive_counts'], 'missing_reports': len(missing)}, ensure_ascii=False))
