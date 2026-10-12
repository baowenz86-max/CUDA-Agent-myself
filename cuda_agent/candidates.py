"""Run-local source snapshots and conservative correctness-based selection."""

import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def source_files(workdir):
    paths = [workdir / 'model_new.py']
    paths += sorted(p for p in (workdir / 'kernels').rglob('*') if p.suffix in {'.cu', '.cpp'})
    return [p for p in paths if p.is_file()]


def invalidate_build(workdir):
    for p in (workdir / 'cuda_extension.so', workdir / '__pycache__'):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    build = workdir / 'build'
    if build.exists():
        shutil.rmtree(build)


def clean_sources(workdir):
    for path in source_files(workdir):
        path.unlink()
    invalidate_build(workdir)


def progress(record):
    verification = record.get('verification') or {}
    rows = verification.get('checks', [])
    fully_passed = bool(record.get('success')) and verification.get('status') == 'passed' and len(rows) == 5 and all(r.get('status') == 'passed' for r in rows)
    passed = sum(r.get('status') == 'passed' for r in rows)
    level = {'sources': 0, 'compile': 1, 'interface': 2, 'verification': 3}.get(record['stage'], 0)
    # Initialization and runtime failures are weaker evidence than value comparisons.
    if record['stage'] == 'verification' and (not rows or any(r.get('stage') in {'reference', 'forward'} for r in rows)):
        level = 2
    return (fully_passed, passed, level)


def better(candidate, best):
    if best is None:
        return True
    if progress(candidate) != progress(best):
        return progress(candidate) > progress(best)
    cv, bv = candidate.get('verification') or {}, best.get('verification') or {}
    if cv.get('seed') != bv.get('seed'):
        return False
    crows = {r['index']: r for r in cv.get('checks', [])}
    brows = {r['index']: r for r in bv.get('checks', [])}
    # Missing observations or newly failing cases cannot improve a candidate.
    if crows.keys() != brows.keys() or any(r.get('status') == 'passed' and crows[i].get('status') != 'passed' for i, r in brows.items()):
        return False
    improvements = False
    for i, previous in brows.items():
        current = crows[i]
        if previous.get('status') == 'passed':
            continue
        ce, be = current.get('error', {}), previous.get('error', {})
        if ce.get('kind') != 'tensor_values' or be.get('kind') != 'tensor_values':
            return False
        if ce.get('path') != be.get('path') or ce.get('expected') != be.get('expected') or current.get('seed') != previous.get('seed'):
            return False
        def score(e):
            return (e.get('mismatch_ratio', 1), float(e.get('max_absolute_error', math.inf)))
        if score(ce) > score(be):
            return False
        improvements |= score(ce) < score(be)
    return improvements


class CandidateStore:
    def __init__(self, workdir, directory):
        self.workdir = Path(workdir)
        self.directory = Path(directory)
        (self.directory / 'candidates').mkdir(parents=True, exist_ok=True)
        self.records = []
        self.best = None
        self.last_sources = {}

    def capture(self, stage, success, log, verification=None, diagnostic=None):
        index = len(self.records) + 1
        destination = self.directory / 'candidates' / f'{index:04d}'
        destination.mkdir()
        sources = {str(p.relative_to(self.workdir)): p.read_text(encoding='utf-8') for p in source_files(self.workdir)}
        manifest = {}
        diffs = []
        for name in sorted(self.last_sources.keys() | sources.keys()):
            diffs.extend(difflib.unified_diff(self.last_sources.get(name, '').splitlines(True), sources.get(name, '').splitlines(True), fromfile='before/' + name, tofile='after/' + name))
        for name, source in sources.items():
            target = destination / 'sources' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding='utf-8')
            manifest[name] = hashlib.sha256(source.encode()).hexdigest()
        self.last_sources = sources
        for name in ("reference_fixture.pt", "failure_inputs.pt"):
            artifact = self.directory / name
            if artifact.exists() and (name == "reference_fixture.pt" or (verification or {}).get("first_failure")):
                # Fixtures are replaced atomically, so existing snapshot hardlinks remain immutable.
                try:
                    os.link(artifact, destination / name)
                except OSError:
                    shutil.copy2(artifact, destination / name)
        for name in ("source_checks.json", "interface.json", "task_contract.json"):
            artifact = self.directory / name
            if artifact.exists():
                shutil.copy2(artifact, destination / name)
        for name, data in (("verification.json", verification), ("diagnostic.json", diagnostic)):
            if data is not None:
                (destination / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        api_path = self.directory / 'api_calls.jsonl'
        api_lines = api_path.read_text(encoding='utf-8').splitlines() if api_path.exists() else []
        record = {'id': index, 'stage': stage, 'success': success, 'manifest': manifest,
                  'verification': verification, 'diagnostic': diagnostic,
                  'api_analysis': json.loads(api_lines[-1]).get('analysis') if api_lines else None,
                  'error': log[-12000:] if not success else '', 'snapshot': str(destination),
                  'replay_inputs': str(destination / 'failure_inputs.pt') if (destination / 'failure_inputs.pt').exists() else None}
        accepted = better(record, self.best)
        record['decision'] = 'best' if accepted else 'rejected'
        record['previous_best'] = self.best['id'] if self.best else None
        if accepted:
            self.best = record
        record['best_candidate'] = self.best['id']
        (destination / 'record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        (destination / 'diff.patch').write_text(''.join(diffs), encoding='utf-8')
        (destination / 'command.log').write_text(log, encoding='utf-8')
        self.records.append(record)
        self._write_context()
        return record

    @staticmethod
    def _summary(record):
        summary = {k: record[k] for k in ('id', 'stage', 'success', 'decision', 'best_candidate', 'api_analysis')}
        verification = record.get('verification') or {}
        summary['verification'] = {
            'seed': verification.get('seed'), 'status': verification.get('status'),
            'checks': [{k: v for k, v in row.items() if k != 'traceback'} for row in verification.get('checks', [])],
            'error': verification.get('error'),
        }
        summary['diagnostic'] = {k: v for k, v in (record.get('diagnostic') or {}).items() if k not in ('traceback', 'sanitizer_log')}
        summary['error_excerpt'] = record['error'][-3000:]
        return summary

    def _write_context(self):
        with (self.directory / 'history.jsonl').open('w', encoding='utf-8') as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        recent = self.records[-6:]
        fingerprints = [json.dumps((r.get('verification') or {}).get('first_failure', {}).get('error', r['error'][-2000:]), sort_keys=True) for r in recent[-2:]]
        stagnant = len(recent) >= 2 and (fingerprints[0] == fingerprints[1] or all(r['decision'] == 'rejected' for r in recent[-2:]))
        context = {'best_candidate': self._summary(self.best), 'recent_candidates': [self._summary(r) for r in recent],
                   'different_hypothesis_required': stagnant,
                   'instruction': 'Diagnose the recorded evidence. If two attempts did not improve, propose a different testable cause. Rejected changes have been rolled back.'}
        (self.directory / 'repair_context.json').write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding='utf-8')

    def restore_best(self):
        if self.best is None:
            return
        clean_sources(self.workdir)
        snapshot = Path(self.best['snapshot']) / 'sources'
        for name in self.best['manifest']:
            target = self.workdir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot / name, target)
        self.last_sources = {name: (self.workdir / name).read_text(encoding='utf-8') for name in self.best['manifest']}

    def best_failure(self):
        if self.best is None:
            return 'No candidate completed evaluation.'
        verification = self.best.get('verification') or {}
        failure = verification.get('first_failure', {}).get('error') or verification.get('error')
        detail = json.dumps(failure, ensure_ascii=False) if failure else self.best['error'][-3000:]
        return f'Best candidate {self.best["id"]}, stage {self.best["stage"]}: {detail}'
