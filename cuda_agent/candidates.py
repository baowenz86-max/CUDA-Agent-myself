"""Run-local source snapshots and conservative correctness-based selection."""

import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from .repair import error_fingerprint


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def source_files(workdir):
    paths = [workdir / 'model_new.py']
    paths += sorted(p for p in (workdir / 'kernels').rglob('*') if p.suffix in {'.cu', '.cpp'})
    return [p for p in paths if p.is_file()]


def invalidate_build(workdir):
    # Native artifacts are validated by compile.py; keep Ninja's incremental cache.
    for p in (workdir / '__pycache__',):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


def clean_sources(workdir):
    for path in source_files(workdir):
        path.unlink()
    invalidate_build(workdir)


def progress(record):
    verification = record.get('verification') or {}
    rows = verification.get('checks', [])
    fully_passed = bool(record.get('success')) and verification.get('status') == 'passed' and len(rows) == 5 and all(r.get('status') == 'passed' for r in rows)
    passed = sum(r.get('status') == 'passed' for r in rows)
    level = {'sources': 0, 'compile': 1, 'interface': 2, 'verification': 4}.get(record['stage'], 0)
    # Initialization and runtime failures are weaker evidence than value comparisons.
    if record['stage'] == 'verification' and (not rows or any(r.get('stage') in {'reference', 'forward'} for r in rows)):
        level = 3
    return (fully_passed, passed, level)


def better(candidate, best):
    if best is None:
        return True
    cv, bv = candidate.get('verification') or {}, best.get('verification') or {}
    if cv and bv and cv.get('seed') != bv.get('seed'):
        return False
    crows = {r['index']: r for r in cv.get('checks', [])}
    brows = {r['index']: r for r in bv.get('checks', [])}
    # Missing observations or newly failing cases cannot improve a candidate.
    if not brows.keys() <= crows.keys() or any(
        current.get('seed') != previous.get('seed') or
        (previous.get('status') == 'passed' and current.get('status') != 'passed')
        for i, previous in brows.items() for current in [crows[i]]
    ):
        return False
    # Checkpoint progress is comparable only on the same saved failing input.
    cd, bd = candidate.get('diagnostic') or {}, best.get('diagnostic') or {}
    cd = cd.get('comparison', cd)
    comparable = (bd.get('status') in {'failed', 'matched'} and cd.get('status') in {'failed', 'matched'}
                  and bd.get('input_seed') is not None
                  and bd.get('input_seed') == cd.get('input_seed'))
    if bd.get('status') in {'failed', 'matched'} and not progress(candidate)[0]:
        if not comparable:
            return False
        if cd.get('matched_checkpoints', 0) < bd.get('matched_checkpoints', 0):
            return False
    if progress(candidate) != progress(best):
        return progress(candidate) > progress(best)
    if not crows or crows.keys() != brows.keys():
        return False
    improvements = False
    for i, previous in brows.items():
        current = crows[i]
        if previous.get('status') == 'passed':
            continue
        if current.get('status') == 'passed':
            improvements = True
            continue
        ce, be = current.get('error', {}), previous.get('error', {})
        levels = {'runtime': 0, 'type': 1, 'length': 1, 'keys': 1,
                  'tensor_structure': 2, 'tensor_values': 3}
        cl, bl = levels.get(ce.get('kind')), levels.get(be.get('kind'))
        if cl is None or bl is None or cl < bl:
            return False
        if cl > bl:
            old_path, new_path = be.get('path'), ce.get('path')
            if old_path and (not new_path or not (new_path == old_path or new_path.startswith(old_path + '['))):
                return False
            # A corrected container can reveal a failure at a child path.
            # Shared seed/observations and no checkpoint regression are required.
            if old_path == new_path and be.get('expected') != ce.get('expected'):
                return False
            improvements = True
            continue
        if ce.get('path') != be.get('path') or ce.get('expected') != be.get('expected'):
            if comparable and cd.get('matched_checkpoints', 0) > bd.get('matched_checkpoints', 0):
                improvements = True
                continue
            return False
        if ce.get('kind') != 'tensor_values' or be.get('kind') != 'tensor_values':
            # Advancing checkpoints is useful even when the next error has the same kind.
            if not comparable or cd.get('matched_checkpoints', 0) <= bd.get('matched_checkpoints', 0):
                return False
            improvements = True
            continue
        def score(e):
            return (e.get('mismatch_ratio', 1), float(e.get('max_absolute_error', math.inf)))
        if score(ce) > score(be):
            return False
        improvements |= score(ce) < score(be)
    return improvements or (comparable and cd.get('matched_checkpoints', 0) > bd.get('matched_checkpoints', 0))


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
        for name in ("source_checks.json", "interface.json", "task_contract.json", "custom_ops_contract.json"):
            artifact = self.directory / name
            if artifact.exists():
                shutil.copy2(artifact, destination / name)
        for name, data in (("verification.json", verification), ("diagnostic.json", diagnostic)):
            if data is not None:
                (destination / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        api_path = self.directory / 'api_calls.jsonl'
        api_lines = api_path.read_text(encoding='utf-8').splitlines() if api_path.exists() else []
        last_call = json.loads(api_lines[-1]) if api_lines else {}
        record = {'id': index, 'stage': stage, 'success': success, 'manifest': manifest,
                  'verification': verification, 'diagnostic': diagnostic,
                  'api_analysis': last_call.get('analysis'), 'api_stage': last_call.get('stage'),
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
        summary = {k: record[k] for k in ('id', 'stage', 'success', 'decision', 'best_candidate', 'api_analysis', 'manifest', 'api_stage')}
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
        fingerprints = [error_fingerprint(r) for r in recent[-2:]]
        stagnant = len(recent) >= 2 and all(r['decision'] == 'rejected' for r in recent[-2:])
        context = {'best_candidate': self._summary(self.best), 'recent_candidates': [self._summary(r) for r in recent],
                   'base_candidate_id': self.best['id'], 'base_manifest': self.best['manifest'],
                   'recent_failure_fingerprints': fingerprints,
                   'different_hypothesis_required': stagnant,
                   'instruction': 'Diagnose the recorded evidence. If two attempts did not improve, propose a different testable cause. Rejected changes have been rolled back.'}
        (self.directory / 'repair_context.json').write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding='utf-8')

    def restore_best(self):
        if self.best is None:
            return
        invalidate_build(self.workdir)
        for path in source_files(self.workdir):
            if str(path.relative_to(self.workdir)) not in self.best['manifest']:
                path.unlink()
        snapshot = Path(self.best['snapshot']) / 'sources'
        for name in self.best['manifest']:
            target = self.workdir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = snapshot / name
            content = source.read_bytes()
            if not target.exists() or target.read_bytes() != content:
                target.write_bytes(content)
        self.last_sources = {name: (self.workdir / name).read_text(encoding='utf-8') for name in self.best['manifest']}
        # Restore evidence as well as sources. A rejected candidate's report must
        # never describe the version the next repair actually edits.
        for name in ('source_checks.json', 'interface.json', 'verification.json',
                     'diagnostic.json', 'failure_inputs.pt', 'custom_ops_contract.json'):
            target = self.directory / name
            artifact = snapshot.parent / name
            if artifact.exists():
                temporary = target.with_suffix(target.suffix + '.restore')
                shutil.copy2(artifact, temporary)
                temporary.replace(target)  # preserve immutable snapshot hardlinks
            else:
                target.unlink(missing_ok=True)
        self._write_context()

    def best_failure(self):
        if self.best is None:
            return 'No candidate completed evaluation.'
        verification = self.best.get('verification') or {}
        failure = verification.get('first_failure', {}).get('error') or verification.get('error')
        detail = json.dumps(failure, ensure_ascii=False) if failure else self.best['error'][-3000:]
        return f'Best candidate {self.best["id"]}, stage {self.best["stage"]}: {detail}'
