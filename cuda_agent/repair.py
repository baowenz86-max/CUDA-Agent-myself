"""Version-bound, category-specific repair evidence."""

import json
import re


TERMINAL_FAILURES = {'reference', 'environment', 'resource'}


def failure_error(record):
    verification = record.get('verification') or {}
    return (verification.get('first_failure', {}).get('error')
            or next((x.get('error', {}) for x in verification.get('checks', []) if x.get('status') == 'failed'), {})
            or verification.get('error') or {})


def failure_category(record):
    stage = record.get('stage', 'unknown')
    if record.get('error', '').startswith('Generation failed'):
        return 'generation'
    verification = record.get('verification') or {}
    error = failure_error(record)
    explicit = record.get('failure_type') or verification.get('failure_type')
    if explicit:
        return explicit
    text = (str(error.get('message', '')) + '\n' + record.get('error', '')).lower()
    if any(x in text for x in ('out of memory', 'cuda_error_out_of_memory', 'std::bad_alloc', 'cannot allocate memory')):
        return 'resource'
    if any(x in text for x in ('no module named \'torch', 'no cuda gpus', 'cuda driver', 'cuda is not available',
                               'cudnn.h: no such file', 'cannot find -lcudnn', 'cudnn development dependency',
                               'nvcc: not found', 'cuda_home environment variable is not set')):
        return 'environment'
    if re.search(r'(?:cudnn\w*\.h.*no such file|no such file.*(?:nvcc|libcudnn))', text):
        return 'environment'
    if stage in TERMINAL_FAILURES:
        return stage
    if any(r.get('stage') == 'reference' and r.get('status') == 'failed' for r in verification.get('checks', [])):
        return 'reference'
    if stage == 'sources' and any(token in text for token in ('contract', 'py::arg', 'c++ type/nullable', 'does not match parameters')):
        return 'contract'
    if stage in {'sources', 'compile', 'interface'}:
        return stage
    if any(token in text for token in ('incompatible function arguments', 'missing required positional argument', 'unexpected keyword argument')):
        return 'binding'
    if any(x in text for x in ('illegal memory', 'launch failure', 'out of bounds', 'misaligned address')):
        return 'runtime'
    if error.get('kind') == 'runtime':
        return 'runtime'
    if error.get('kind') in {'type', 'length', 'keys', 'tensor_structure'}:
        return 'shape'
    if error.get('kind') in {'tensor_values', 'scalar'}:
        return 'numerical'
    return 'unknown'


def normalize_message(message):
    message = re.sub(r'0x[0-9a-fA-F]+', '<address>', str(message))
    message = re.sub(r'(?:/[\w.+-]+)+/([\w.+-]+)', r'\1', message)
    message = re.sub(r'([\w.-]+\.(?:cu|cpp|py)):\d+(?::\d+)?', r'\1:<location>', message)
    return re.sub(r'\s+', ' ', message).strip()[:1000]


def error_fingerprint(record):
    error = failure_error(record)
    diagnostic = record.get('diagnostic') or {}
    checkpoint = next((x.get('id') for x in diagnostic.get('checkpoints', [])
                       if x.get('trustworthy') is True and x.get('matched') is False), None)
    raw_message = error.get('message', record.get('error', ''))
    if not error and record.get('stage') in {'compile', 'sources', 'interface'}:
        root_errors = re.findall(r'(?im)^.*?(?:fatal error:|error:|(?:RuntimeError|TypeError|ValueError):).+$', raw_message)
        if root_errors:
            raw_message = root_errors[0]
    message = normalize_message(raw_message)
    exception = re.search(r'\b([A-Za-z_]+(?:Error|Exception))\b', message)
    return json.dumps({'stage': record.get('stage'), 'category': failure_category(record),
                       'kind': error.get('kind'), 'path': error.get('path'), 'checkpoint': checkpoint,
                       'exception': exception.group(1) if exception else None,
                       'expected': error.get('expected'), 'actual': error.get('actual'),
                       'message': message}, sort_keys=True)


def repair_effect(base, candidate, analysis=None):
    from .candidates import compare, progress
    before, after = base.get('manifest', {}), candidate.get('manifest', {})
    changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    added, deleted = sorted(after.keys() - before.keys()), sorted(before.keys() - after.keys())
    targets = (analysis or {}).get('target_files', [])
    unapplied = sorted(k for k in targets if before.get(k) == after.get(k))
    improved, reason = compare(candidate, base)
    terminal = failure_category(candidate) in TERMINAL_FAILURES
    reverse, _ = compare(base, candidate) if not terminal else (False, 'terminal_failure')
    same = error_fingerprint(base) == error_fingerprint(candidate)
    outcome = ('fixed' if progress(candidate)['correctness'] else 'progressed' if improved else
               'regressed' if reverse or 'regression' in reason else 'same_failure' if same else 'different_failure')
    return {'base_candidate_id': base.get('id'), 'candidate_id': candidate.get('id'),
            'before_hashes': before, 'after_hashes': after, 'changed_files': changed,
            'added_files': added, 'deleted_files': deleted, 'unapplied_target_files': unapplied,
            'repair_not_applied': bool(unapplied) or not (changed or added or deleted),
            'before_progress': progress(base), 'after_progress': progress(candidate),
            'before_fingerprint': error_fingerprint(base), 'after_fingerprint': error_fingerprint(candidate),
            'outcome': outcome, 'terminal_failure': terminal, 'same_fingerprint': same, 'decision_reason': reason,
            'hypothesis': (analysis or {}).get('hypothesis')}


def concise_log(log):
    """Surface root errors and signatures; leave full command logs on disk."""
    lines = log.splitlines()
    selected = []
    for i, line in enumerate(lines):
        if re.search(r'error:|undefined (?:symbol|reference)|Error:|Missing key|Unexpected key|incompatible function|no visible|prohibited|contract|AssertionError|ERROR SUMMARY|Invalid (?:__global__|read|write)', line, re.I):
            selected.extend(lines[i:i + 4])
    return '\n'.join(dict.fromkeys(selected))[:4000] if selected else log[-2000:]
