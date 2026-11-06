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


def diagnostic_evidence(record):
    diagnostic = record.get('diagnostic') or {}
    return diagnostic.get('comparison', diagnostic)


def trusted_prefix(diagnostic):
    rows = diagnostic.get('checkpoints', [])
    count = 0
    for row in rows:
        if row.get('trustworthy') is not True or row.get('matched') is not True:
            break
        count += 1
    return count


def progress(record):
    verification = record.get('verification') or {}
    rows = verification.get('checks', [])
    complete = (bool(record.get('success')) and verification.get('status') == 'passed'
                and len(rows) == 5 and all(r.get('status') == 'passed' for r in rows))
    level = {'generation': 0, 'sources': 0, 'compile': 1, 'interface': 2,
             'runtime': 3, 'verification': 4}.get(record.get('stage'), 0)
    if record.get('stage') == 'verification' and (not rows or any(
            r.get('stage') in {'reference', 'forward'} for r in rows)):
        level = 3
    diagnostic = diagnostic_evidence(record)
    return {'correctness': complete, 'pipeline': level,
            'passed_checks': sum(r.get('status') == 'passed' for r in rows),
            'trusted_matched': trusted_prefix(diagnostic)}


def _score(error):
    return (error.get('mismatch_ratio', 1), float(error.get('max_absolute_error', math.inf)))


def compare(candidate, best):
    """Return a stable decision reason; absent trust is never inferred from counts."""
    from .repair import failure_category, TERMINAL_FAILURES
    if failure_category(candidate) in TERMINAL_FAILURES:
        return False, 'terminal_failure'
    if best is None:
        return True, 'initial_candidate'
    cp, bp = progress(candidate), progress(best)
    cv, bv = candidate.get('verification') or {}, best.get('verification') or {}
    if cv and bv and cv.get('seed') != bv.get('seed'):
        return False, 'verification_seed_changed'
    crows = {r['index']: r for r in cv.get('checks', [])}
    brows = {r['index']: r for r in bv.get('checks', [])}
    if not brows.keys() <= crows.keys():
        return False, 'missing_normal_observations'
    if any(crows[i].get('seed') != r.get('seed') or
           (r.get('status') == 'passed' and crows[i].get('status') != 'passed')
           for i, r in brows.items()):
        return False, 'normal_observation_regression'
    if cp['correctness'] != bp['correctness']:
        return cp['correctness'], 'correctness_success' if cp['correctness'] else 'correctness_regression'
    cd, bd = diagnostic_evidence(candidate), diagnostic_evidence(best)
    comparable = (bd.get('input_seed') is not None and bd.get('input_seed') == cd.get('input_seed')
                  and bd.get('fixture_id') is not None and bd.get('fixture_id') == cd.get('fixture_id')
                  and bool(bd.get('reference_sequence'))
                  and bd.get('reference_sequence') == cd.get('reference_sequence'))
    # A missing or unreliable diagnostic is neutral. A confirmed regression is not.
    cchecks, bchecks = cd.get('checkpoints', []), bd.get('checkpoints', [])
    if comparable:
        for i, row in enumerate(bchecks[:bp['trusted_matched']]):
            if i < len(cchecks) and cchecks[i].get('trustworthy') is True and cchecks[i].get('matched') is False:
                return False, 'checkpoint_regression'
    if cp['pipeline'] != bp['pipeline']:
        return cp['pipeline'] > bp['pipeline'], 'pipeline_progress' if cp['pipeline'] > bp['pipeline'] else 'pipeline_regression'
    if comparable:
        if cp['trusted_matched'] > bp['trusted_matched']:
            return True, 'checkpoint_progress'
        # Missing evidence cannot establish a shorter matched prefix as a regression.
        n = bp['trusted_matched']
        if n < len(bchecks) and n < len(cchecks):
            c, b = cchecks[n], bchecks[n]
            if (c.get('trustworthy') is True and b.get('trustworthy') is True
                    and c.get('id') == b.get('id') and c.get('matched') is False and b.get('matched') is False
                    and c.get('reference') == b.get('reference')):
                if _score(c.get('mismatch') or {}) != _score(b.get('mismatch') or {}):
                    win = _score(c.get('mismatch') or {}) < _score(b.get('mismatch') or {})
                    return win, 'checkpoint_mismatch_progress' if win else 'checkpoint_mismatch_regression'
    # Different known input/fixture/sequence cannot establish checkpoint progress.
    if any(bd.get(key) is not None and cd.get(key) is not None and bd[key] != cd[key]
           for key in ('input_seed', 'fixture_id', 'reference_sequence')):
        return False, 'incomparable_diagnostic'
    improvements = False
    levels = {'runtime': 0, 'type': 1, 'length': 1, 'keys': 1,
              'tensor_structure': 2, 'tensor_values': 3, 'scalar': 3}
    for i, previous in brows.items():
        current = crows[i]
        if previous.get('status') == 'passed':
            continue
        if current.get('status') == 'passed':
            improvements = True
            continue
        ce, be = current.get('error', {}), previous.get('error', {})
        cl, bl = levels.get(ce.get('kind')), levels.get(be.get('kind'))
        if cl is None or bl is None or cl < bl:
            return False, 'final_evidence_regression_or_unknown'
        if cl > bl:
            old, new = be.get('path'), ce.get('path')
            if old and (not new or not (new == old or new.startswith(old + '['))):
                return False, 'incomparable_output_path'
            if old == new and be.get('expected') != ce.get('expected'):
                return False, 'reference_metadata_changed'
            improvements = True
            continue
        if ce.get('path') != be.get('path') or ce.get('expected') != be.get('expected'):
            return False, 'incomparable_output'
        if ce.get('kind') not in {'tensor_values', 'scalar'}:
            continue
        if _score(ce) > _score(be):
            return False, 'final_mismatch_regression'
        improvements |= _score(ce) < _score(be)
    return improvements, 'final_progress' if improvements else 'tie_keep_previous'


def trusted_advancement(candidate, best):
    """Independent audit predicate: never consult the comparator decision."""
    from .repair import failure_category, TERMINAL_FAILURES
    if best is None or failure_category(candidate) in TERMINAL_FAILURES:
        return False
    cp, bp = progress(candidate), progress(best)
    cv, bv = candidate.get('verification') or {}, best.get('verification') or {}
    crows = {r['index']:r for r in cv.get('checks', [])}
    brows = {r['index']:r for r in bv.get('checks', [])}
    if (cv and bv and cv.get('seed') != bv.get('seed')) or not brows.keys() <= crows.keys():
        return False
    if any(crows[i].get('seed') != r.get('seed') or r.get('status') == 'passed' and crows[i].get('status') != 'passed' for i,r in brows.items()):
        return False
    cd, bd = diagnostic_evidence(candidate), diagnostic_evidence(best)
    comparable = (bd.get('fixture_id') is not None and bd.get('fixture_id') == cd.get('fixture_id')
                  and bd.get('input_seed') is not None and bd.get('input_seed') == cd.get('input_seed')
                  and bool(bd.get('reference_sequence')) and bd.get('reference_sequence') == cd.get('reference_sequence'))
    if comparable:
        cchecks = cd.get('checkpoints', [])
        if any(i < len(cchecks) and cchecks[i].get('trustworthy') is True and cchecks[i].get('matched') is False for i in range(bp['trusted_matched'])):
            return False
    return ((cp['correctness'] and not bp['correctness']) or cp['pipeline'] > bp['pipeline'] or
            comparable and cp['pipeline'] >= bp['pipeline'] and cp['trusted_matched'] > bp['trusted_matched'])


def better(candidate, best):
    return compare(candidate, best)[0]


class CandidateStore:
    def __init__(self, workdir, directory):
        self.workdir = Path(workdir)
        self.directory = Path(directory)
        (self.directory / 'candidates').mkdir(parents=True, exist_ok=True)
        self.records = []
        self.best = None
        self.last_sources = {}
        self.repair_base = None

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
        for name in ("source_checks.json", "interface.json", "interface_precompile.json", "task_contract.json", "custom_ops_contract.json", "dependencies.json"):
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
        record['schema_version'] = 2
        record['progress'] = progress(record)
        record['fingerprint'] = error_fingerprint(record)
        accepted, record['decision_reason'] = compare(record, self.best)
        base = self.repair_base
        if base is not None:
            from .repair import repair_effect
            record['repair'] = repair_effect(base, record, last_call.get('analysis'))
            with (self.directory / 'repair_effects.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record['repair'], ensure_ascii=False) + '\n')
            self.repair_base = None
        record['decision'] = 'best' if accepted else 'rejected'
        record['previous_best'] = self.best['id'] if self.best else None
        if accepted:
            self.best = record
        record['best_candidate'] = self.best['id'] if self.best else None
        (destination / 'record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        (destination / 'diff.patch').write_text(''.join(diffs), encoding='utf-8')
        (destination / 'command.log').write_text(log, encoding='utf-8')
        self.records.append(record)
        self._write_context()
        return record

    @staticmethod
    def _summary(record):
        if record is None:
            return None
        summary = {k: record[k] for k in ('id', 'stage', 'success', 'decision', 'best_candidate', 'api_analysis', 'manifest', 'api_stage')}
        summary.update({k: record.get(k) for k in ('progress', 'fingerprint', 'decision_reason', 'repair')})
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
        effect_path = self.directory / 'repair_effects.jsonl'
        effects = [json.loads(line) for line in effect_path.read_text().splitlines()] if effect_path.exists() else []
        repeated_effects = len(effects) >= 2 and all(e.get('outcome') == 'same_failure' or e.get('repair_not_applied') for e in effects[-2:])
        stagnant = repeated_effects or len(recent) >= 2 and all(r['decision'] == 'rejected' for r in recent[-2:])
        context = {'best_candidate': self._summary(self.best), 'recent_candidates': [self._summary(r) for r in recent],
                   'base_candidate_id': self.best['id'] if self.best else None, 'base_manifest': self.best['manifest'] if self.best else {},
                   'repair_effects': effects[-6:],
                   'recent_failure_fingerprints': fingerprints,
                   'different_hypothesis_required': stagnant,
                   'instruction': 'Recheck the root cause, repeated fingerprints, unapplied target files and previous hypotheses. Separate observed facts, mathematical inference and unverified hypotheses; large digamma output does not establish large input. Diagnose the recorded evidence. If two attempts did not improve, propose a different testable cause. Rejected changes have been rolled back.'}
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
        for name in ('source_checks.json', 'interface.json', 'interface_precompile.json', 'dependencies.json', 'verification.json',
                     'diagnostic.json', 'failure_inputs.pt', 'reference_fixture.pt', 'task_contract.json', 'custom_ops_contract.json'):
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
